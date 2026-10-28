# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.v41_layerwise import (
    is_v41_model,
    select_v41_block_pairs,
    strided_transfer_ranges,
    validate_v41_layerwise_config,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    KVCacheSendingLayerThread,
    LayerMetadata,
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
    SendTask,
)


def make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v41")),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        speculative_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False, swa_bounded_replay=False),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
    )


def test_flattened_model_config_and_supported_graph_mode():
    config = make_config()
    assert is_v41_model(config)
    validate_v41_layerwise_config(config)
    config.model_config.hf_text_config.model_type = "deepseek_v41_text"
    assert is_v41_model(config)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "use_ubatching", True),
        ("cache_config", "enable_prefix_caching", True),
        ("cache_config", "swa_bounded_replay", True),
        ("compilation_config", "cudagraph_mode", CUDAGraphMode.FULL),
    ],
)
def test_unsupported_execution_fails_before_cache_registration(section, field, value):
    config = make_config()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match="V4.1"):
        validate_v41_layerwise_config(config)


def test_speculative_state_requires_a_separate_transfer_contract():
    config = make_config()
    config.speculative_config = SimpleNamespace(method="mtp")
    with pytest.raises(ValueError, match="MTP/DSpark"):
        validate_v41_layerwise_config(config)


@pytest.mark.parametrize(
    "tokens,start", [(127, 0), (128, 0), (129, 0), (142, 0), (143, 1), (255, 8), (256, 8), (257, 8), (271, 9)]
)
def test_swa_uses_absolute_positions_and_retains_only_live_window(tokens, start):
    spec = SimpleNamespace(block_size=16, sliding_window=128)
    end = (tokens + 15) // 16
    local = [0] * start + list(range(50 + start, 50 + end))
    remote = [0] * start + list(range(100 + start, 100 + end))
    assert select_v41_block_pairs(local, remote, spec, tokens) == (local[start:], remote[start:])


def test_c2_state_is_one_request_page_even_for_long_prompt():
    spec = CircularBufferSpec(block_size=32, num_kv_heads=1, head_size=1024, dtype=torch.float32, head_size_v=0)
    assert select_v41_block_pairs([7], [17], spec, 8193) == ([7], [17])
    with pytest.raises(ValueError, match="one state page"):
        select_v41_block_pairs([7, 8], [17], spec, 8193)


def test_compression_does_not_divide_logical_block_ids():
    spec = SimpleNamespace(block_size=16, sliding_window=None, tokens_per_state=2)
    assert select_v41_block_pairs([7, 8, 9], [17, 18, 19], spec, 33) == ([7, 8, 9], [17, 18, 19])
    with pytest.raises(ValueError, match="null or invalid"):
        select_v41_block_pairs([7, 0, 9], [17, 18, 19], spec, 33)


def test_page_strided_planes_copy_without_touching_other_page_regions():
    source = torch.arange(8 * 64, dtype=torch.int64).to(torch.uint8).reshape(8, 64)
    destination = torch.full((9, 96), 211, dtype=torch.uint8)
    # Long KV, index K and scale occupy different regions of each page.
    offsets, sizes = [0, 20, 28], [20, 8, 2]
    local = LayerMetadata([0] * 3, [source.data_ptr() + o for o in offsets], sizes, [1] * 3, [64] * 3, "v41")
    remote = LayerMetadata([0] * 3, [destination.data_ptr() + o for o in offsets], sizes, [1] * 3, [96] * 3, "v41")
    src, dst, lengths = strided_transfer_ranges([2, 3, 6], [4, 5, 1], local, remote)
    for address, target, length in zip(src, dst, lengths):
        ctypes.memmove(target, address, length)
    expected = torch.full_like(destination, 211)
    for left, right in zip([2, 3, 6], [4, 5, 1]):
        for offset, size in zip(offsets, sizes):
            expected[right, offset : offset + size] = source[left, offset : offset + size]
    torch.testing.assert_close(destination, expected)


def test_layout_mismatch_rejected_before_transfer():
    local = LayerMetadata([0], [1000], [16], [1], [32], "v41")
    remote = LayerMetadata([0], [2000], [16], [1])
    with pytest.raises(ValueError, match="protocol versions"):
        strided_transfer_ranges([1], [2], local, remote)


def test_sender_uses_each_components_own_group():
    sender = KVCacheSendingLayerThread.__new__(KVCacheSendingLayerThread)
    sender.pd_head_ratio = 1
    sender.layer_metadata = {
        "swa": LayerMetadata([2], [1000], [8], [1], [64], "swa"),
        "long": LayerMetadata([0], [2000], [16], [1], [64], "long"),
        "state": LayerMetadata([1], [3000], [32], [1], [64], "state"),
    }
    sender.kv_cache_specs = [None] * 3
    sender.engine = MagicMock()
    sender.engine.batch_transfer_sync_write.return_value = 0
    sender.failed_reqs = set()
    req = SimpleNamespace(
        remote_host="peer",
        remote_te_rpc_port=123,
        local_block_ids=[[2], [3], [4]],
        remote_block_ids=[[5], [6], [7]],
        remote_layer_metadata=sender.layer_metadata,
    )
    task = SendTask(
        send_request={"req": req}, wait_event=MagicMock(), layer_name="swa", layer_names=["swa", "long", "state"]
    )
    assert sender._transfer_kv_cache(task) is None
    _, src, dst, lengths = sender.engine.batch_transfer_sync_write.call_args.args
    assert src == [1000 + 4 * 64, 2000 + 2 * 64, 3000 + 3 * 64]
    assert dst == [1000 + 7 * 64, 2000 + 5 * 64, 3000 + 6 * 64]
    assert lengths == [8, 16, 32]


def test_final_chunk_snapshot_and_rank_mapping():
    worker = MooncakeLayerwiseConnectorWorker.__new__(MooncakeLayerwiseConnectorWorker)
    worker.tp_size, worker.tp_rank = 2, 1
    worker.num_kv_cache_groups = 1
    worker.block_size = [16]
    worker.kv_cache_specs = [SimpleNamespace(block_size=16, sliding_window=None)]
    request = SimpleNamespace(
        chunk_finish=True,
        remote_tp_size=2,
        remote_pcp_size=1,
        remote_dcp_size=1,
        remote_block_size=[16],
        local_block_ids=[[2, 3]],
        remote_block_ids=[[7, 8]],
        local_computed_tokens=31,
        remote_port=1234,
    )
    metadata = MooncakeLayerwiseConnectorMetadata()
    metadata.requests = {"partial": SimpleNamespace(chunk_finish=False), "final": request}
    worker._prepare_v41_step(metadata)
    assert set(metadata.requests) == {"final"}
    assert metadata.requests["final"].remote_port == 1235
    assert metadata.requests["final"].trans_count == [1]
    assert request.remote_port == 1234


def test_state_handoff_leaves_last_prompt_token_for_decode():
    scheduler = MooncakeLayerwiseConnectorScheduler.__new__(MooncakeLayerwiseConnectorScheduler)
    scheduler._is_v41 = scheduler.need_truncate = True
    assert scheduler._hybrid_prefill_token_count(129) == 128
    # The ring group must survive even when prompt length is 1 mod 32.
    blocks = ([1, 2, 3, 4, 5], [9])
    assert scheduler._trim_hybrid_remote_block_ids(blocks, 129) is blocks


def test_missing_layer_hook_notifies_decode_failure_instead_of_success():
    worker = MooncakeLayerwiseConnectorWorker.__new__(MooncakeLayerwiseConnectorWorker)
    worker.current_layer, worker.total_layers = 2, 40
    worker.send_done_send_signal = MagicMock()
    request = object()
    with pytest.raises(RuntimeError, match="2 of 40"):
        worker.check_v41_step_complete(SimpleNamespace(requests={"req": request}))
    worker.send_done_send_signal.assert_called_once_with("req", request, 0, trans_flag=False)


def test_publish_records_event_and_enqueues_once_then_drain_protects_pages():
    connector = MooncakeLayerwiseConnector.__new__(MooncakeLayerwiseConnector)
    connector._is_v41 = connector._is_kv_producer = True
    connector.has_connector_metadata = lambda: True
    connector._connector_metadata = SimpleNamespace(requests={"req": object()})
    connector.connector_worker = MagicMock()
    connector.save_v41_kv_layer("model.layers.2.self_attn.swa_cache")
    connector.connector_worker.on_kv_cache_written.assert_called_once()
    connector.connector_worker.save_kv_layer.assert_called_once()
    connector.wait_for_save()
    connector.connector_worker.kv_send_layer_thread.send_queue.join.assert_called_once()


def test_publication_requires_an_explicit_connector_capability(monkeypatch):
    monkeypatch.setattr(dsa_v41, "has_kv_transfer_group", lambda: True)
    generic_save = MagicMock()
    monkeypatch.setattr(dsa_v41, "get_kv_transfer_group", lambda: SimpleNamespace(save_kv_layer=generic_save))
    dsa_v41.publish_v41_kv_layer("swa")
    generic_save.assert_not_called()
    publish = MagicMock()
    monkeypatch.setattr(dsa_v41, "get_kv_transfer_group", lambda: SimpleNamespace(save_v41_kv_layer=publish))
    dsa_v41.publish_v41_kv_layer("swa")
    publish.assert_called_once_with("swa")


@pytest.mark.parametrize("profile", [False, True])
def test_layerwise_publication_follows_cache_writes_on_empty_query_rank(monkeypatch, profile):
    impl = AscendDSAV41Impl.__new__(AscendDSAV41Impl)
    impl.swa_prefix = "model.layers.2.self_attn.swa_cache"
    metadata = SimpleNamespace(swa=SimpleNamespace(num_actual_tokens=0))
    events = []
    monkeypatch.setattr(dsa_v41, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None if profile else {}))
    monkeypatch.setattr(dsa_v41, "oproj_tp_enable", lambda: False)
    monkeypatch.setattr(dsa_v41, "publish_v41_kv_layer", lambda name: events.append(("publish", name)))
    impl._get_layer_metadata = lambda _: metadata

    def prepare(*args):
        events.append(("cache", impl.swa_prefix))
        return None, None, None

    impl._prepare_inputs_and_caches = prepare
    impl._project_output = lambda *args, **kwargs: events.append(("output", impl.swa_prefix))
    projector = SimpleNamespace(enable_pcp_o_proj_weight_sharding=False)
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=projector)), n_local_heads=1, head_dim=8
    )
    impl.forward(attn, None, torch.empty((0, 8)))
    assert events == ([] if profile else [(stage, impl.swa_prefix) for stage in ("cache", "publish", "output")])
