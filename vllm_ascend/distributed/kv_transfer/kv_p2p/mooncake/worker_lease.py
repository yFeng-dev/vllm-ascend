# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Worker-owned leases for V1/Hybrid, using the shared heartbeat sender."""

import threading
import time
from collections import OrderedDict
from typing import Any

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.heartbeat import (
    DEFAULT_KV_LEASE_DURATION,
    MooncakeHeartbeatThread,
)


def get_kv_lease_duration(kv_transfer_config: Any) -> int:
    duration = kv_transfer_config.get_from_extra_config("kv_lease_duration", DEFAULT_KV_LEASE_DURATION)
    if type(duration) is not int or duration < 6:
        raise ValueError("Mooncake kv_lease_duration must be an integer of at least 6 seconds")
    return duration


class KVCacheTaskTracker:
    """Worker completion tracking with renewable, locally timed producer leases."""

    def __init__(self, kv_lease_duration: int = DEFAULT_KV_LEASE_DURATION):
        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()
        self.reqs_to_process: set[str] = set()
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.renewed_requests: OrderedDict[str, float] = OrderedDict()
        self.kv_lease_duration = kv_lease_duration
        self.lease_extension = kv_lease_duration * 2 // 3

    def add_req_to_process(self, request_id: str):
        with self.done_task_lock:
            self.reqs_to_process.add(request_id)

    def _finish_locked(self, request_id: str):
        self.reqs_to_process.discard(request_id)
        self.delayed_free_requests.pop(request_id, None)
        self.renewed_requests.pop(request_id, None)
        self.finished_requests.add(request_id)

    def add_not_transfer_request(self, request_id: str):
        with self.done_task_lock:
            self._finish_locked(request_id)

    def update_done_task_count(self, request_id: str, *, report_completion: bool = True):
        with self.done_task_lock:
            if not report_completion:
                # Notification-only requests never entered WAITING_FOR_REMOTE_KVS.
                self.reqs_to_process.discard(request_id)
                self.delayed_free_requests.pop(request_id, None)
                self.renewed_requests.pop(request_id, None)
                return
            if request_id in self.reqs_to_process:
                self._finish_locked(request_id)

    def add_delayed_request(self, request_id: str):
        with self.done_task_lock:
            if (
                request_id in self.reqs_to_process
                and request_id not in self.delayed_free_requests
                and request_id not in self.renewed_requests
            ):
                # Scheduler timestamps cannot be compared across P worker hosts.
                self.delayed_free_requests[request_id] = time.monotonic() + self.kv_lease_duration

    def renew_requests(self, request_ids: list[str] | tuple[str, ...]):
        with self.done_task_lock:
            now = time.monotonic()
            for request_id in request_ids:
                expiry = self.delayed_free_requests.get(request_id, self.renewed_requests.get(request_id))
                if expiry is None:
                    continue
                if now >= expiry:
                    self._finish_locked(request_id)
                    logger.error("Force freed expired Mooncake request %s after KV lease expired", request_id)
                elif now + self.lease_extension > expiry:
                    self.delayed_free_requests.pop(request_id, None)
                    self.renewed_requests[request_id] = now + self.lease_extension
                    self.renewed_requests.move_to_end(request_id)

    def _retrieve_expired_requests(self) -> set[str]:
        """Called under done_task_lock; both queues remain ordered by expiry."""
        now = time.monotonic()
        expired = set()
        for requests in (self.delayed_free_requests, self.renewed_requests):
            while requests:
                request_id, expiry = next(iter(requests.items()))
                if now < expiry:
                    break
                self._finish_locked(request_id)
                expired.add(request_id)
                logger.error("Force freed expired Mooncake request %s after KV lease expired", request_id)
        return expired

    def get_and_clear_finished_requests(self) -> set[str]:
        with self.done_task_lock:
            self._retrieve_expired_requests()
            finished = self.finished_requests.copy()
            self.finished_requests.clear()
            return finished


class MooncakeLeaseScheduler:
    """Shared V1/hybrid scheduler hooks; data and DONE still use worker ports."""

    def _init_kv_leases(self) -> None:
        self._kv_lease_duration = get_kv_lease_duration(self.vllm_config.kv_transfer_config)
        self._heartbeat_thread: MooncakeHeartbeatThread | None = None
        self._recv_without_load: set[str] = set()
        self._heartbeat_request_counts: dict[str, int] = {}

    def _lease_transfer_params(self) -> dict[str, Any]:
        parallel = self.vllm_config.parallel_config
        count = self.tp_size * self.pcp_size * parallel.pipeline_parallel_size
        base_port = self.vllm_config.kv_transfer_config.kv_port
        endpoints = []
        for port in range(self.side_channel_port, self.side_channel_port + count):
            peer = self.multi_nodes_meta_mapping.get(str(port - base_port), {})
            endpoints.append((peer.get("host", self.side_channel_host), peer.get("handshake_port", port)))
        return {
            "remote_kv_lease_duration": self._kv_lease_duration,
            "remote_kv_lease_endpoints": endpoints,
        }

    def on_new_request(self, request: Any) -> None:
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_prefill") or not any(params.get("remote_block_ids", ())):
            return
        remote_duration = params.get("remote_kv_lease_duration")
        if type(remote_duration) is not int or remote_duration != self._kv_lease_duration:
            raise ValueError(
                "Mooncake requires matching integer kv_lease_duration on P and D; "
                f"P advertised {remote_duration!r}, D uses {self._kv_lease_duration}. "
                "Upgrade P before D and preserve kv_transfer_params through the proxy."
            )
        endpoints = params.get("remote_kv_lease_endpoints")
        if not isinstance(endpoints, (tuple, list)) or not endpoints:
            raise ValueError("Mooncake producer must provide remote_kv_lease_endpoints")
        peers = []
        for endpoint in endpoints:
            if (
                not isinstance(endpoint, (tuple, list))
                or len(endpoint) != 2
                or not isinstance(endpoint[0], str)
                or not endpoint[0]
                or type(endpoint[1]) is not int
                or not 0 < endpoint[1] < 65536
            ):
                raise ValueError(f"Invalid Mooncake lease endpoint: {endpoint!r}")
            peers.append((endpoint[0], endpoint[1], params["remote_request_id"]))
        if self._heartbeat_thread is None:
            self._heartbeat_thread = MooncakeHeartbeatThread(self._kv_lease_duration // 6)
            self._heartbeat_thread.start()
        # V2 renews one scheduler endpoint; V1/Hybrid renew each worker.
        # The final numeric component keeps these local registry keys unique
        # even when a request ID itself contains colons. Wire IDs stay unchanged.
        self._stop_renewing(request.request_id)
        for index, (host, port, remote_id) in enumerate(peers):
            self._heartbeat_thread.add_request(f"{request.request_id}:{index}", host, port, remote_id)
        self._heartbeat_request_counts[request.request_id] = len(peers)

    def update_connector_output(self, connector_output: Any) -> None:
        finished = connector_output.finished_recving or set()
        if self._heartbeat_thread is not None:
            for request_id in finished:
                self._stop_renewing(request_id)
        # Empty receives only notify P to release its KV. A cache-hit request
        # is already running, and a cancelled request may already be removed
        # from vLLM's scheduler; neither is WAITING_FOR_REMOTE_KVS.
        no_load_finished = self._recv_without_load.intersection(finished)
        if no_load_finished:
            finished.difference_update(no_load_finished)
            self._recv_without_load.difference_update(no_load_finished)

    def _finish_without_load(self, request_id: str) -> None:
        self._stop_renewing(request_id)
        self._recv_without_load.add(request_id)

    def _stop_renewing(self, request_id: str) -> None:
        count = self._heartbeat_request_counts.pop(request_id, 0)
        if self._heartbeat_thread is not None:
            for index in range(count):
                self._heartbeat_thread.remove_request(f"{request_id}:{index}")

    def shutdown(self) -> None:
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.shutdown()
            self._heartbeat_thread = None
        self._recv_without_load.clear()
        self._heartbeat_request_counts.clear()
