# SPDX-License-Identifier: Apache-2.0
"""Materialize the shared QSA six-region planner's views for Model Runner V2."""

import math
from collections.abc import Callable

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm_ascend.core.six_region_kv_cache_layout import (
    GDN,
    HIDDEN,
    PLE,
    QSA_COMPRESSED,
    QSA_MAIN,
    QSA_RAW,
    build_six_region_kv_cache_layout,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers
from vllm_ascend.worker.utils import make_page_strided_cache_view


def allocate_six_region_cache(
    config: KVCacheConfig,
    allocate: Callable[[int], torch.Tensor],
) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]] | None:
    """Honor split K/V and conv/SSM slabs without importing or invoking Runner V1."""
    layout = build_six_region_kv_cache_layout(config.kv_cache_groups, config.num_blocks)
    if layout is None:
        return None
    size = layout.slot_count * layout.slot_backing_size
    descriptors = {
        name: (descriptor, index)
        for descriptor in config.kv_cache_tensors
        for index, name in enumerate(get_kv_cache_tensor_layers(descriptor))
    }
    # A descriptor anchors the first region; its stride is not the combined
    # logical page size of a two-component QSA or GDN owner.
    for owner in layout.owners:
        if owner.role == HIDDEN:
            continue
        descriptor, index = descriptors[owner.layer_name]
        region = layout.region(layout.region_name_for_owner(owner.layer_name))
        if (
            descriptor.size != size
            or descriptor.offset + index * descriptor.layer_stride
            != owner.slot * layout.slot_backing_size + region.offset
            or descriptor.block_stride != region.page_size_bytes
        ):
            raise ValueError(f"Invalid six-region descriptor for {owner.layer_name}")
    backing = allocate(size)
    caches = {}
    for owner in layout.owners:
        spec = owner.spec
        if owner.role == HIDDEN:
            raw = allocate(config.num_blocks * spec.page_size_bytes)
            caches[owner.layer_name] = make_page_strided_cache_view(
                raw,
                (
                    config.num_blocks,
                    spec.num_heads,
                    spec.num_states,
                    spec.state_content_size_bytes // get_dtype_size(spec.dtype),
                ),
                spec.dtype,
                spec.page_size_bytes,
            )
            continue
        if owner.role == QSA_MAIN:
            components = (
                ("r2", (spec.block_size, spec.num_kv_heads, spec.head_size), spec.dtype),
                ("r3", (spec.block_size, spec.num_kv_heads, spec.head_size_v), spec.dtype),
            )
        elif owner.role in (QSA_RAW, QSA_COMPRESSED):
            components = (
                (
                    "r4" if owner.role == QSA_RAW else "r5",
                    (spec.num_heads, spec.num_states, spec.state_content_size_bytes // get_dtype_size(spec.dtype)),
                    spec.dtype,
                ),
            )
        elif owner.role in (GDN, PLE):
            regions = ("r1", "r2") if owner.role == GDN else ("r6",)
            components = tuple(zip(regions, spec.shapes, spec.dtypes, strict=True))
        else:
            raise ValueError(f"Unsupported six-region owner {owner}")
        views = []
        for region_name, shape, dtype in components:
            region = layout.region(region_name)
            item_bytes = math.prod(shape) * get_dtype_size(dtype)
            if item_bytes > region.page_size_bytes:
                raise ValueError(f"Six-region payload exceeds its page for {owner.layer_name}")
            views.append(
                make_page_strided_cache_view(
                    backing,
                    (config.num_blocks, *shape),
                    dtype,
                    region.page_size_bytes,
                    owner.slot * layout.slot_backing_size + region.offset,
                )
            )
        if owner.role == QSA_MAIN:
            caches[owner.layer_name] = tuple(views)
        elif owner.role in (GDN, PLE):
            caches[owner.layer_name] = views
        else:
            caches[owner.layer_name] = views[0]
    return caches
