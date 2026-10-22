# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_qwen4_exp_six_region_kv_cache_config,
)
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.six_region_cache import allocate_six_region_cache
from vllm_ascend.worker.worker import NPUWorker


def configuration(tp, block_size=None):
    block_size = block_size or 768 * 8 // tp
    prefix = "model.layers.3.self_attn"
    specs = {
        prefix + ".attn": FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.bfloat16,
        ),
        prefix + ".indexer.raw_key_cache": CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.float32,
        ),
        prefix + ".indexer.compressed_key_cache": MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            tokens_per_state=4,
        ),
        "model.layers.0.linear_attn": MambaSpec(
            block_size=block_size,
            shapes=((3, 80 * 128 // tp), (48 // tp, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
        ),
        "model.ple": MambaSpec(
            block_size=block_size,
            shapes=((2, 10240),),
            dtypes=(torch.bfloat16,),
            tp_replicated=True,
        ),
    }
    groups = [KVCacheGroupSpec([name], spec) for name, spec in specs.items()]
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=3, prefix_cache_retention_interval=None)
    )
    return _get_qwen4_exp_six_region_kv_cache_config(cfg, groups, 16 * 1024 * 1024)


@pytest.mark.parametrize("tp,block_size", [(4, 1536), (8, 768), (8, 1536)])
def test_v2_materializes_planned_six_regions(monkeypatch, tp, block_size):
    config = configuration(tp, block_size)
    layers = {
        name: SimpleNamespace(kv_sharing_target_layer_name=None)
        for g in config.kv_cache_groups
        for name in g.layer_names
    }
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: SimpleNamespace(kv_transfer_config=None))
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *a, **k: layers)
    caches = attn_utils.allocate_kv_cache_main(config, torch.device("cpu"), None, [])
    prefix = "model.layers.3.self_attn"
    key, value = caches[prefix + ".attn"]
    conv, ssm = caches["model.layers.0.linear_attn"]
    raw = caches[prefix + ".indexer.raw_key_cache"]
    compressed = caches[prefix + ".indexer.compressed_key_cache"]
    (ple,) = caches["model.ple"]
    assert key.shape == value.shape == (3, block_size, 1, 256)
    assert raw.shape == (3, 1, 4, 128)
    assert compressed.shape == (3, 1, block_size // 4, 128)
    assert ssm.dtype == raw.dtype == torch.float32
    assert key.dtype == conv.dtype == ple.dtype == torch.bfloat16
    views = (key, value, conv, ssm, raw, compressed, ple)
    assert len({t.untyped_storage().data_ptr() for t in views}) == 1
    assert all(t.is_contiguous() for t in (key, value, conv, raw, compressed, ple))
    assert key.data_ptr() == ssm.data_ptr()  # Deliberate cross-group alias.
    key[0].fill_(7)
    ssm[1].fill_(3)
    payload_tokens = ssm[1].numel() * ssm.element_size() // (256 * 2)
    assert torch.all(key[1, payload_tokens:] == 0)  # SSM padding is untouched.
    assert torch.all(key[0] == 7)
    assert torch.all(key[2] == 0)
    for index, tensor in enumerate((value, conv, raw, compressed, ple), 1):
        tensor.fill_(index)
    assert torch.all(key[0] == 7) and torch.all(ssm[1] == 3)
    for index, tensor in enumerate((value, conv, raw, compressed, ple), 1):
        assert torch.all(tensor == index)


def test_six_region_rejects_corrupt_descriptor_before_allocating():
    config = configuration(8)
    config.kv_cache_tensors[0].offset += 16
    with pytest.raises(ValueError, match="Invalid six-region descriptor"):
        allocate_six_region_cache(config, lambda _: pytest.fail("must validate before allocation"))


@pytest.mark.parametrize("tp", [4, 8])
def test_six_region_budget_counts_physical_backing_once(monkeypatch, tp):
    config = configuration(tp, 1536)
    worker = NPUWorker.__new__(NPUWorker)
    worker.model_runner = SimpleNamespace(supports_six_region_kv_backing=True)
    worker.vllm_config = SimpleNamespace()
    worker.get_kv_cache_spec = lambda: {g.layer_names[0]: g.kv_cache_spec for g in config.kv_cache_groups}
    monkeypatch.setattr("vllm_ascend.worker.worker.get_kv_cache_groups", lambda *a: config.kv_cache_groups)
    assert worker._scale_kv_cache_memory_for_multi_group(1024 * 1024 * 1024) == 1024 * 1024 * 1024
