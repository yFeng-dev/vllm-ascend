# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for descriptor-only QSA backends on Model Runner V2."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm_ascend.ops.gbsa_qsa import _as_gbsa_page_pool
from vllm_ascend.worker.v2 import attn_utils


class DescriptorBackend:
    uses_descriptor_cache_view = True

    @classmethod
    def full_cls_name(cls):
        return cls.__name__

    @classmethod
    def is_sparse(cls):
        return False


class QSAMainBackend(DescriptorBackend):
    uses_qsa_page_major_kv_view = True


class StateBackend(DescriptorBackend):
    uses_descriptor_cache_view = False


@pytest.mark.parametrize("kind", ["main", "compressed", "ring"])
@pytest.mark.parametrize("padding", [0, 4096])
@pytest.mark.parametrize("hybrid", [False, True])
def test_qsa_descriptor_views_through_v2_allocator(monkeypatch, kind, padding, hybrid):
    if kind == "main":
        spec = FullAttentionSpec(
            block_size=768,
            num_kv_heads=2,
            head_size=256,
            dtype=torch.bfloat16,
            num_head_slots=2,
            state_content_bytes=2 * 256 * 2,
        )
    elif kind == "compressed":
        spec = MLAAttentionSpec(
            block_size=768,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            tokens_per_state=4,
        )
    else:
        spec = CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.float32,
        )
    if kind == "main" and padding:
        padding = 128 * 2 * 256 * 2
    spec = replace(spec, page_size_padded=spec.page_size_bytes + padding)
    name = "model.layers.3.qsa"
    blocks = 3
    config = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=blocks * spec.page_size_bytes,
                layers=[name],
                layer_stride=blocks * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
                offset=0,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)],
    )
    backend = QSAMainBackend if kind == "main" else DescriptorBackend
    layer = SimpleNamespace(get_attn_backend=lambda: backend, kv_sharing_target_layer_name=None, num_heads=24)
    layers = {name: layer}
    if hybrid:
        state_name = "model.layers.0.linear_attn"
        state_spec = MambaSpec(
            block_size=768,
            shapes=((3, 8), (2, 2, 2)),
            dtypes=(torch.float32, torch.float32),
            page_size_padded=spec.page_size_bytes,
        )
        layer_bytes = blocks * spec.page_size_bytes
        config.kv_cache_tensors = [
            KVCacheTensor(
                size=2 * layer_bytes,
                layers=[layer_name],
                layer_stride=layer_bytes,
                block_stride=spec.page_size_bytes,
                offset=index * layer_bytes,
            )
            for index, layer_name in enumerate((state_name, name))
        ]
        config.kv_cache_groups.append(KVCacheGroupSpec(layer_names=[state_name], kv_cache_spec=state_spec))
        layers[state_name] = SimpleNamespace(
            get_attn_backend=lambda: StateBackend,
            kv_sharing_target_layer_name=None,
            num_heads=0,
        )
    vllm_config = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(cache_dtype="auto"),
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        quant_config=None,
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *a, **k: layers)
    monkeypatch.setattr(attn_utils, "requires_contiguous_pa_kv_cache", lambda *a, **k: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda *a, **k: False)
    monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda *a, **k: False)
    caches = attn_utils.allocate_kv_cache_main(
        config,
        device=torch.device("cpu"),
        layout=None,
        kernel_block_sizes=[128 if kind != "ring" else 4, 768] if hybrid else [128 if kind != "ring" else 4],
    )
    result = caches[name]
    if kind == "main":
        assert isinstance(result, tuple) and len(result) == 2
        key, value = result
        assert key.shape == value.shape == (blocks, 768, 2, 256)
        assert key.stride()[1:] == (512, 256, 1)
        assert value.data_ptr() - key.data_ptr() == 768 * 2 * 256 * 2
        key.fill_(7)
        value.fill_(11)
        assert torch.all(key == 7) and torch.all(value == 11)
        for tensor in (key, value):
            pool, block_stride, subpage_stride, subpages = _as_gbsa_page_pool(tensor)
            assert subpages == 6 and subpage_stride == 1
            for block in range(blocks):
                assert torch.equal(
                    pool[block * block_stride : block * block_stride + subpages].reshape(768, 2, 256), tensor[block]
                )
        views = result
    else:
        assert isinstance(result, torch.Tensor)
        assert result.shape == (blocks, 1, spec.num_states, 128)
        views = (result,)
    for view in views:
        assert view.stride(0) * view.element_size() == spec.page_size_bytes
    if hybrid:
        state = caches[state_name][0]
        assert state.untyped_storage().data_ptr() == views[0].untyped_storage().data_ptr()
        assert views[0].data_ptr() - state.data_ptr() == layer_bytes
        before = [view.clone() for view in views]
        state.fill_(99)
        for view, expected in zip(views, before):
            assert torch.equal(view, expected)
