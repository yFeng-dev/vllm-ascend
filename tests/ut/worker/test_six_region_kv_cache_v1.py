# SPDX-License-Identifier: Apache-2.0
"""CPU checks against the actual Runner V1 allocator and view construction."""

from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.patch.platform.patch_kv_cache_utils import _get_qwen4_exp_six_region_kv_cache_config
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker


def check(block_size):
    """Exercise allocation, actual writes, strides, aliases and budget together."""
    prefix = "model.layers.3.self_attn"
    specs = {
        prefix + ".attn": FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=256, dtype=torch.bfloat16),
        prefix + ".indexer.raw_key_cache": CircularBufferSpec(
            block_size=4, num_kv_heads=1, head_size=128, head_size_v=0, dtype=torch.float32
        ),
        prefix + ".indexer.compressed_key_cache": MLAAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, tokens_per_state=4
        ),
        "model.layers.0.linear_attn": MambaSpec(
            block_size=block_size, shapes=((3, 1280), (6, 128, 128)), dtypes=(torch.bfloat16, torch.float32)
        ),
        "model.ple": MambaSpec(
            block_size=block_size, shapes=((2, 10240),), dtypes=(torch.bfloat16,), tp_replicated=True
        ),
    }
    groups = [KVCacheGroupSpec([name], spec) for name, spec in specs.items()]
    cfg = NS(cache_config=NS(num_gpu_blocks_override=3, prefix_cache_retention_interval=None))
    config = _get_qwen4_exp_six_region_kv_cache_config(cfg, groups, 16 * 1024 * 1024)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.ascend_config = NS(kvpp_config=NS(size=1), xlite_graph_config=NS(enabled=False))
    for key in (
        "use_sparse",
        "enable_sparse_sfa_c8",
        "enable_sparse_li_c8",
        "use_compress",
        "use_hybrid_blocks",
        "hybrid_with_attn_and_mamba",
        "is_kv_consumer",
        "sparse_kv_offload_enabled",
    ):
        setattr(runner, key, False)
    runner.sfa_dcp_replicated_indexer_size = runner.dcp_size = 1
    runner.runner_only_attn_layers = set()
    runner.sparse_kv_offload_config = MagicMock()
    runner.tp_rank = 0
    runner.compilation_config = NS(static_forward_context={})
    runner.vllm_config = MagicMock()
    runner.vllm_config.kv_transfer_config = NS(kv_connector="MooncakeConnectorV2")
    runner.model_config = MagicMock()
    runner.model_config.use_mla = True
    runner.dtype = torch.bfloat16
    runner.attn_groups = []
    runner.kv_cache_config = config
    backend = MagicMock()
    backend.is_sparse.return_value = False
    runner.attn_backend = backend
    runner._kv_cache_spec_attn_group_iterator = lambda: [
        NS(kv_cache_group_id=i, kv_cache_spec=g.kv_cache_spec, backend=backend, layer_names=g.layer_names)
        for i, g in enumerate(config.kv_cache_groups)
    ]
    with patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config", return_value={}):
        backing = runner._allocate_kv_cache_tensors(config)
        caches = runner._reshape_kv_cache_tensors(config, backing)
    layout = runner._six_region_kv_cache_layout
    assert layout is not None and len(layout.regions) == 6
    assert len({id(t) for t in backing.values()}) == 1
    slab = next(iter(backing.values()))
    assert slab.data_ptr() % (2 * 1024 * 1024) == 0
    slab.zero_()
    key, value = caches[prefix + ".attn"]
    conv, ssm = caches["model.layers.0.linear_attn"]
    raw = caches[prefix + ".indexer.raw_key_cache"]
    compressed = caches[prefix + ".indexer.compressed_key_cache"]
    (ple,) = caches["model.ple"]
    assert key.shape == value.shape == (3, block_size, 1, 256)
    assert raw.shape == (3, 1, 4, 128)
    assert compressed.shape == (3, 1, block_size // 4, 128)
    assert key.data_ptr() == ssm.data_ptr()
    assert len({t.untyped_storage().data_ptr() for t in (key, value, conv, ssm, raw, compressed, ple)}) == 1
    for tensor, region in (
        (key, "r2"),
        (value, "r3"),
        (conv, "r1"),
        (ssm, "r2"),
        (raw, "r4"),
        (compressed, "r5"),
        (ple, "r6"),
    ):
        assert tensor.stride(0) * tensor.element_size() == layout.region(region).page_size_bytes
        assert tensor.data_ptr() == slab.data_ptr() + layout.region(region).offset
    key[0].fill_(7)
    ssm[1].fill_(3)
    payload = ssm[1].numel() * ssm.element_size() // 512
    assert torch.all(key[1, payload:] == 0)
    assert torch.all(key[0] == 7) and torch.all(key[2] == 0)
    for index, tensor in enumerate((value, conv, raw, compressed, ple), 1):
        tensor.fill_(index)
    assert torch.all(key[0] == 7) and torch.all(ssm[1] == 3)
    for index, tensor in enumerate((value, conv, raw, compressed, ple), 1):
        assert torch.all(tensor == index)
    # Explicit sliced-storage offsets must survive view construction.
    shifted = torch.zeros(slab.numel() + 256, dtype=torch.int8)[256:]
    owner = layout.owner(prefix + ".attn")
    shifted_key, _ = runner._make_six_region_layer_views(layout, owner, shifted)
    assert shifted_key.data_ptr() == shifted.data_ptr() + layout.region("r2").offset
    worker = NPUWorker.__new__(NPUWorker)
    worker.model_runner = runner
    worker.vllm_config = runner.vllm_config
    worker.get_kv_cache_spec = lambda: specs
    with patch("vllm_ascend.worker.worker.get_kv_cache_groups", return_value=config.kv_cache_groups):
        assert worker._scale_kv_cache_memory_for_multi_group(1024**3) == 1024**3
    runner.vllm_config.kv_transfer_config = NS(kv_connector="OtherConnector")
    assert not runner.supports_six_region_kv_backing
    print(
        f"PASS TP8 block_size={block_size}: allocator, six-region strides, aliases, "
        "storage offset, memory budget, connector gate",
        flush=True,
    )


@pytest.mark.parametrize("block_size", [768, 1536])
def test_runner_v1_six_region_tp8(block_size):
    check(block_size)
