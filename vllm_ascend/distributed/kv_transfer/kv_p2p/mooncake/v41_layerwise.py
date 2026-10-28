# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-owned V4.1 cache pages for the layerwise push connector."""

from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec


def is_v41_model(config) -> bool:
    return config.model_config.hf_text_config.model_type in ("deepseek_v41", "deepseek_v41_text")


def validate_v41_layerwise_config(config) -> None:
    parallel = config.parallel_config
    if (
        any(
            size != 1
            for size in (
                parallel.pipeline_parallel_size,
                parallel.prefill_context_parallel_size,
                parallel.decode_context_parallel_size,
            )
        )
        or parallel.use_ubatching
    ):
        raise ValueError("V4.1 Mooncake layerwise requires PP=PCP=DCP=1 and no microbatching")
    if config.speculative_config is not None:
        raise ValueError("V4.1 Mooncake layerwise does not yet transfer MTP/DSpark prompt state")
    if config.cache_config.enable_prefix_caching or getattr(config.cache_config, "swa_bounded_replay", False):
        raise ValueError("V4.1 Mooncake layerwise requires prefix caching and SWA bounded replay disabled")
    if config.compilation_config.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY):
        raise ValueError("V4.1 Mooncake layerwise prefill requires NONE or FULL_DECODE_ONLY graph mode")


def group_specs(group_spec):
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        return tuple(group_spec.kv_cache_specs.values())
    return (group_spec,)


def select_v41_block_pairs(local_ids, remote_ids, spec, computed_tokens):
    """Pair original-token block coordinates, or the request's circular page.

    Compression changes page payloads, not logical block IDs. SWA block lists
    retain evicted-prefix placeholders; never compact them before pairing.
    """
    specs = group_specs(spec)
    if all(is_circular_kv_cache_spec(item) for item in specs):
        if len(local_ids) != 1 or len(remote_ids) != 1:
            raise ValueError("V4.1 C2 transfer requires one state page per request on both peers")
        start, end = 0, 1
    else:
        block_size = spec.block_size
        end = (computed_tokens + block_size - 1) // block_size
        windows = {getattr(item, "sliding_window", None) for item in specs}
        if len(windows) != 1:
            raise ValueError("V4.1 transfer group contains incompatible sliding windows")
        window = windows.pop()
        start = max(0, computed_tokens - window + 1) // block_size if window else 0
        if end > len(local_ids) or end > len(remote_ids):
            raise ValueError("V4.1 transfer block table does not cover the computed prefix")
    local, remote = list(local_ids[start:end]), list(remote_ids[start:end])
    if any(block <= 0 for block in (*local, *remote)):
        raise ValueError("V4.1 transfer selected a null or invalid cache block")
    return local, remote


def strided_transfer_ranges(local_ids, remote_ids, local_meta, remote_meta):
    """Copy component payloads without overwriting other views of a page."""
    if len(local_ids) != len(remote_ids):
        raise ValueError("V4.1 source and destination block counts differ")
    if not local_meta.layout or local_meta.layout != remote_meta.layout:
        raise ValueError("V4.1 Mooncake peers must use matching cache layouts and protocol versions")
    components = len(local_meta.block_len)
    if any(
        len(values) != components
        for values in (
            local_meta.kv_caches_base_addr,
            remote_meta.kv_caches_base_addr,
            remote_meta.block_len,
            local_meta.block_stride,
            remote_meta.block_stride,
        )
    ):
        raise ValueError("V4.1 cache component count mismatch")
    src, dst, lengths = [], [], []
    for index, payload in enumerate(local_meta.block_len):
        if payload != remote_meta.block_len[index]:
            raise ValueError("V4.1 cache payload size mismatch")
        src_stride, dst_stride = local_meta.block_stride[index], remote_meta.block_stride[index]
        if min(src_stride, dst_stride) < payload:
            raise ValueError("V4.1 cache payload exceeds page stride")
        for local, remote in zip(local_ids, remote_ids):
            src_addr = local_meta.kv_caches_base_addr[index] + local * src_stride
            dst_addr = remote_meta.kv_caches_base_addr[index] + remote * dst_stride
            if src and src_addr == src[-1] + lengths[-1] and dst_addr == dst[-1] + lengths[-1]:
                lengths[-1] += payload
            else:
                src.append(src_addr)
                dst.append(dst_addr)
                lengths.append(payload)
    return src, dst, lengths
