# SPDX-License-Identifier: Apache-2.0
"""QSA views for Runner V1's per-layer private cache allocations."""

import math

import torch


def make_private_qsa_views(runner, raw, spec, blocks, backend):
    """Keep K/V token rows dense while retaining the physical page stride."""
    itemsize = torch.empty((), dtype=spec.dtype).element_size()

    def view(shape, relative_offset=0):
        assert relative_offset + math.prod(shape[1:]) * itemsize <= spec.page_size_bytes
        return runner._page_strided_tensor_view(
            raw,
            dtype=spec.dtype,
            shape=shape,
            block_stride_bytes=spec.page_size_bytes,
            storage_offset_bytes=raw.storage_offset() + relative_offset,
        )

    if getattr(backend, "uses_qsa_page_major_kv_view", False):
        shape = (blocks, spec.block_size, spec.num_kv_heads, spec.head_size)
        slot_bytes = math.prod(shape[1:]) * itemsize
        return view(shape), view(shape, slot_bytes)

    shape = (blocks, spec.num_heads, spec.num_states, spec.state_content_size_bytes // itemsize)
    return view(shape)
