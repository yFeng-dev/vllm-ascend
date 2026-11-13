# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import contextlib
import importlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.heartbeat import (
    HEARTBEAT_MSG,
    MooncakeHeartbeatThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_heartbeat import (
    KVCacheTaskTracker,
    MooncakeLeaseScheduler,
    get_kv_lease_duration,
)


@pytest.mark.parametrize("duration", [6, 7, 120, 600])
def test_config_timing(duration):
    config = SimpleNamespace(get_from_extra_config=lambda key, default: duration)
    assert get_kv_lease_duration(config) == duration
    tracker = KVCacheTaskTracker(duration)
    tracker.add_req_to_process("p")
    with patch("time.monotonic", return_value=100):
        tracker.add_delayed_request("p")
    assert tracker.delayed_free_requests["p"] == 100 + duration
    assert tracker.lease_extension == duration * 2 // 3


@pytest.mark.parametrize("duration", [None, True, 5, 0, -1, 6.0, "120"])
def test_invalid_config(duration):
    with pytest.raises(ValueError, match="integer"):
        get_kv_lease_duration(SimpleNamespace(get_from_extra_config=lambda key, default: duration))


def test_lease_renewal_expiry_and_no_resurrection():
    tracker = KVCacheTaskTracker(120)
    with patch("time.monotonic", return_value=0):
        for request_id in ("renew", "expire", "done"):
            tracker.add_req_to_process(request_id)
            tracker.add_delayed_request(request_id)
    with patch("time.monotonic", return_value=20):
        tracker.renew_requests(["renew"])
    assert tracker.delayed_free_requests["renew"] == 120  # Never shorten initial lease.
    with patch("time.monotonic", return_value=60):
        tracker.renew_requests(["renew"])
        tracker.add_delayed_request("renew")  # Repeated metadata cannot reset the deadline.
    assert tracker.renewed_requests["renew"] == 140
    tracker.update_done_task_count("done")
    with patch("time.monotonic", return_value=120):
        tracker.renew_requests(["expire", "done", "unknown"])
        assert tracker.get_and_clear_finished_requests() == {"expire", "done"}
    with patch("time.monotonic", return_value=140):
        tracker.renew_requests(["renew"])
        assert tracker.get_and_clear_finished_requests() == {"renew"}
        tracker.renew_requests(["renew"])
        assert tracker.get_and_clear_finished_requests() == set()
    assert not tracker.reqs_to_process
    assert not tracker.delayed_free_requests
    assert not tracker.renewed_requests


def test_renewal_order_and_completion_cleanup():
    tracker = KVCacheTaskTracker(6)
    with patch("time.monotonic", return_value=0):
        for request_id in ("a", "b", "c"):
            tracker.add_req_to_process(request_id)
            tracker.add_delayed_request(request_id)
    with patch("time.monotonic", return_value=3):
        tracker.renew_requests(["b", "a"])
    with patch("time.monotonic", return_value=4):
        tracker.renew_requests(["b"])
    assert list(tracker.renewed_requests) == ["a", "b"]
    with patch("time.monotonic", return_value=7):
        assert tracker.get_and_clear_finished_requests() == {"a", "c"}
    tracker.add_not_transfer_request("b")
    assert not tracker.renewed_requests
    assert tracker.get_and_clear_finished_requests() == {"b"}


def test_notification_only_completion_clears_renewed_lease_without_reporting_load():
    tracker = KVCacheTaskTracker(6)
    with patch("time.monotonic", return_value=0):
        tracker.add_req_to_process("notification")
        tracker.add_delayed_request("notification")
    with patch("time.monotonic", return_value=3):
        tracker.renew_requests(["notification"])
    tracker.update_done_task_count("notification", report_completion=False)
    with patch("time.monotonic", return_value=100):
        assert tracker.get_and_clear_finished_requests() == set()
    assert not tracker.reqs_to_process
    assert not tracker.delayed_free_requests
    assert not tracker.renewed_requests


def make_scheduler(cls=MooncakeLeaseScheduler):
    scheduler = object.__new__(cls)
    scheduler.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_port=10000, get_from_extra_config=lambda key, default: default),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )
    scheduler.tp_size = 2
    scheduler.pcp_size = 1
    scheduler.side_channel_port = 10004  # DP rank 1, two PP stages, TP2.
    scheduler.side_channel_host = "127.0.0.1"
    scheduler.multi_nodes_meta_mapping = {
        "6": {"host": "127.0.0.2", "handshake_port": 10006},
        "7": {"host": "127.0.0.2", "handshake_port": 10007},
    }
    scheduler._reqs_need_recv = {}
    scheduler._reqs_in_batch = set()
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[None, None])
    scheduler.kv_cache_groups = [None, None]
    scheduler._init_kv_leases()
    return scheduler


def make_request(scheduler, **overrides):
    params = dict(
        do_remote_prefill=True,
        remote_block_ids=([1], [2]),
        remote_request_id="p",
        remote_engine_id="producer",
        remote_host="127.0.0.1",
        remote_port=10004,
        **scheduler._lease_transfer_params(),
    )
    params.update(overrides)
    return SimpleNamespace(request_id="d", kv_transfer_params=params, status=None)


def test_all_worker_endpoints_and_aggregated_completion():
    scheduler = make_scheduler()
    request = make_request(scheduler)
    assert request.kv_transfer_params["remote_kv_lease_endpoints"] == [
        ("127.0.0.1", 10004),
        ("127.0.0.1", 10005),
        ("127.0.0.2", 10006),
        ("127.0.0.2", 10007),
    ]
    with patch.object(MooncakeHeartbeatThread, "start"):
        scheduler.on_new_request(request)
    heartbeat = scheduler._heartbeat_thread
    assert set(heartbeat.requests.values()) == {
        (host, port, "p") for host, port in request.kv_transfer_params["remote_kv_lease_endpoints"]
    }
    scheduler.update_connector_output(SimpleNamespace(finished_recving=None))
    assert heartbeat.requests
    scheduler.update_connector_output(SimpleNamespace(finished_recving={"d"}))
    assert not heartbeat.requests
    scheduler.shutdown()


def test_worker_registration_replacement_and_completion_keep_other_requests():
    scheduler = make_scheduler()
    first = make_request(scheduler)
    other = make_request(scheduler, remote_request_id="other-producer")
    other.request_id = "d:0"
    with patch.object(MooncakeHeartbeatThread, "start"):
        scheduler.on_new_request(first)
        scheduler.on_new_request(other)
        first.kv_transfer_params["remote_kv_lease_endpoints"] = [("127.0.0.3", 10008)]
        scheduler.on_new_request(first)
    heartbeat = scheduler._heartbeat_thread
    assert len(heartbeat.requests) == 5
    assert ("127.0.0.3", 10008, "p") in heartbeat.requests.values()
    scheduler.update_connector_output(SimpleNamespace(finished_recving={"d"}))
    assert len(heartbeat.requests) == 4
    assert {peer[2] for peer in heartbeat.requests.values()} == {"other-producer"}
    scheduler.update_connector_output(SimpleNamespace(finished_recving={"d:0"}))
    assert not heartbeat.requests
    assert not scheduler._heartbeat_request_counts
    scheduler.shutdown()


@pytest.mark.parametrize("duration", [None, True, "120", 6, 120.0])
def test_remote_mismatch_rejected_before_thread_start(duration):
    scheduler = make_scheduler()
    with pytest.raises(ValueError, match="matching integer"):
        scheduler.on_new_request(make_request(scheduler, remote_kv_lease_duration=duration))
    assert scheduler._heartbeat_thread is None


@pytest.mark.parametrize("endpoints", [None, [], ["host"], [("host", True)], [("host", 0)], [("", 1000)]])
def test_invalid_peer_list_rejected(endpoints):
    scheduler = make_scheduler()
    with pytest.raises(ValueError):
        scheduler.on_new_request(make_request(scheduler, remote_kv_lease_endpoints=endpoints))
    assert scheduler._heartbeat_thread is None


@pytest.mark.parametrize("module_name", ["mooncake_connector", "mooncake_hybrid_connector"])
def test_connector_cancellation_and_hooks(module_name):
    module = importlib.import_module(f"vllm_ascend.distributed.kv_transfer.kv_p2p.{module_name}")
    scheduler = make_scheduler(module.MooncakeConnectorScheduler)
    connector = object.__new__(module.MooncakeConnector)
    connector.connector_scheduler = scheduler
    request = make_request(scheduler)
    with patch.object(MooncakeHeartbeatThread, "start"):
        connector.on_new_request(request)
    heartbeat = scheduler._heartbeat_thread
    assert heartbeat.requests  # Registration precedes allocation.
    assert connector.request_finished_all_groups(request, ([], [])) == (False, None)
    assert not heartbeat.requests
    assert not request.kv_transfer_params["do_remote_prefill"]
    assert "d" in scheduler._reqs_in_batch
    assert len(scheduler._reqs_need_recv) == 1
    connector.request_finished_all_groups(request, ([], []))
    assert len(scheduler._reqs_need_recv) == 1
    # V1 suppresses notification-only completions in its worker; Hybrid filters
    # them when the scheduler receives the aggregated output.
    output = SimpleNamespace(finished_recving={"d"} if module_name == "mooncake_hybrid_connector" else set())
    connector.update_connector_output(output)
    assert not output.finished_recving
    assert not scheduler._recv_without_load
    connector.shutdown()


@pytest.mark.parametrize("module_name", ["mooncake_connector", "mooncake_hybrid_connector"])
def test_full_cache_hit_does_not_report_a_scheduler_load(module_name):
    module = importlib.import_module(f"vllm_ascend.distributed.kv_transfer.kv_p2p.{module_name}")
    scheduler = make_scheduler(module.MooncakeConnectorScheduler)
    request = make_request(scheduler)
    with patch.object(MooncakeHeartbeatThread, "start"):
        scheduler.on_new_request(request)
    scheduler.update_state_after_alloc(request, MagicMock(), 0)
    assert not scheduler._heartbeat_thread.requests
    finished = {"other-actual-load"}
    if module_name == "mooncake_hybrid_connector":
        finished.add("d")
    output = SimpleNamespace(finished_recving=finished)
    scheduler.update_connector_output(output)
    assert output.finished_recving == {"other-actual-load"}
    assert not scheduler._recv_without_load
    scheduler.shutdown()


@pytest.mark.parametrize("module_name", ["mooncake_connector", "mooncake_hybrid_connector"])
def test_producer_control_protocol(module_name):
    module = importlib.import_module(f"vllm_ascend.distributed.kv_transfer.kv_p2p.{module_name}")
    sender = object.__new__(module.KVCacheSendingThread)
    sender.metadata = {}
    sender.task_tracker = KVCacheTaskTracker(6)
    sender.port_send_num = {}
    sender.task_tracker.add_req_to_process("p")
    with patch("time.monotonic", return_value=0):
        sender.add_delayed_request("p")
    messages = [(HEARTBEAT_MSG, ["p"]), (module.GET_META_MSG, ""), (module.DONE_RECVING_MSG, "p", None)]
    sock = MagicMock()
    sock.recv_multipart.side_effect = [[b"identity", b"", msgspec.msgpack.encode(message)] for message in messages] + [
        SystemExit()
    ]
    with patch("time.monotonic", return_value=3), pytest.raises(SystemExit):
        sender.run_busy_loop(sock)
    assert sock.send_multipart.call_count == 3
    assert sender.task_tracker.get_and_clear_finished_requests() == {"p"}
    assert not sender.task_tracker.renewed_requests


@pytest.mark.parametrize("module_name", ["mooncake_connector", "mooncake_hybrid_connector"])
def test_live_worker_leases_survive_waiting_then_release(module_name):
    """Exercise both production ROUTER loops beyond an actual six-second lease."""
    module = importlib.import_module(f"vllm_ascend.distributed.kv_transfer.kv_p2p.{module_name}")
    context = zmq.Context()
    producers = []
    threads = []
    endpoints = []
    scheduler = make_scheduler()
    scheduler._kv_lease_duration = 6

    class StoppableSocket:
        def __init__(self, sock):
            self.sock = sock

        def recv_multipart(self):
            frames = self.sock.recv_multipart()
            if frames[-1] == b"STOP":
                self.sock.send_multipart((frames[0], b"", b"ACK"))
                raise SystemExit
            return frames

        def send_multipart(self, *args, **kwargs):
            return self.sock.send_multipart(*args, **kwargs)

    def serve(sender, ready):
        with context.socket(zmq.ROUTER) as sock:
            port = sock.bind_to_random_port("tcp://127.0.0.1")
            endpoints.append(("127.0.0.1", port))
            ready.set()
            with contextlib.suppress(SystemExit):
                sender.run_busy_loop(StoppableSocket(sock))

    try:
        for _ in range(2):
            sender = object.__new__(module.KVCacheSendingThread)
            sender.metadata = {}
            sender.task_tracker = KVCacheTaskTracker(6)
            sender.port_send_num = {}
            for request_id in ("waiting", "complete"):
                sender.task_tracker.add_req_to_process(request_id)
                sender.add_delayed_request(request_id)
            producers.append(sender)
            ready = threading.Event()
            thread = threading.Thread(target=serve, args=(sender, ready), daemon=True)
            threads.append(thread)
            thread.start()
            assert ready.wait(5)
        for request_id in ("waiting", "complete"):
            request = make_request(
                scheduler,
                remote_request_id=request_id,
                remote_kv_lease_duration=6,
                remote_kv_lease_endpoints=endpoints,
            )
            request.request_id = f"d-{request_id}"
            scheduler.on_new_request(request)
        deadline = time.monotonic() + 7
        while time.monotonic() < deadline:
            for sender in producers:
                assert not sender.get_and_clear_finished_requests()
            time.sleep(0.05)
        # A DONE releases immediately even while the other request renews.
        for host, port in endpoints:
            with context.socket(zmq.REQ) as sock:
                sock.setsockopt(zmq.RCVTIMEO, 2000)
                sock.connect(f"tcp://{host}:{port}")
                sock.send(msgspec.msgpack.encode((module.DONE_RECVING_MSG, "complete", None)))
                assert sock.recv() == b"ACK"
        for sender in producers:
            assert sender.get_and_clear_finished_requests() == {"complete"}
        scheduler.update_connector_output(SimpleNamespace(finished_recving={"d-complete"}))
        assert {peer[2] for peer in scheduler._heartbeat_thread.requests.values()} == {"waiting"}
        scheduler.shutdown()  # D disappears: no further scheduler or network work.
        deadline = time.monotonic() + 6
        expired = [set(), set()]
        while time.monotonic() < deadline and any("waiting" not in result for result in expired):
            for index, sender in enumerate(producers):
                expired[index].update(sender.get_and_clear_finished_requests())
            time.sleep(0.05)
        assert expired == [{"waiting"}, {"waiting"}]
    finally:
        scheduler.shutdown()
        for host, port in endpoints:
            with context.socket(zmq.REQ) as sock:
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVTIMEO, 2000)
                sock.connect(f"tcp://{host}:{port}")
                sock.send(b"STOP")
                sock.recv()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
        context.term()
