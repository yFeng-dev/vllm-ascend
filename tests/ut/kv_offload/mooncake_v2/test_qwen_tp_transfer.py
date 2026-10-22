# SPDX-License-Identifier: Apache-2.0
"""Exercise actual CPU payloads using Mooncake's generated READ addresses."""

import ctypes
import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeTransferMetadataGroups,
)

from .helpers import (
    make_full_spec,
    make_kv_cache_tensor,
    make_pp_metadata,
    make_transfer_metadata,
)
from .test_pull_scheduler import make_sending_thread
from .test_pull_worker import make_thread


def padded_tensor(shape, dtype, padding):
    page_bytes = math.prod(shape[1:]) * torch.tensor([], dtype=dtype).element_size()
    stride = page_bytes + padding
    raw = torch.full((shape[0] * stride,), 0xA5, dtype=torch.uint8)
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    strides[0] = stride // torch.tensor([], dtype=dtype).element_size()
    view = torch.as_strided(raw.view(dtype), shape, strides)
    return raw, view


def cache_fields(caches):
    return dict(
        block_shapes=[[tuple(t.shape[1:]) for t in caches]],
        block_strides=[[t.stride(0) * t.element_size() for t in caches]],
        block_lens=[[math.prod(t.shape[1:]) * t.element_size() for t in caches]],
    )


def execute_read(thread, remote, remote_tp, remote_rank, local_raw, remote_raw):
    local_addresses, remote_addresses, sizes = [], [], []
    thread._append_spec_transfer_addresses(
        spec_index=0,
        remote_pcp_rank=0,
        remote_tp_rank=remote_rank,
        remote_pcp_size=1,
        remote_tp_size=remote_tp,
        remote_dcp_size=1,
        transfer_entries_by_layer={(0, 0): [("request", [2, 0], [1, 3])]},
        remote_metadata=remote,
        src_list=local_addresses,
        dst_list=remote_addresses,
        length_list=sizes,
    )
    for local, source, size in zip(local_addresses, remote_addresses, sizes):
        assert any(t.data_ptr() <= local and local + size <= t.data_ptr() + t.nbytes for t in local_raw)
        assert any(t.data_ptr() <= source and source + size <= t.data_ptr() + t.nbytes for t in remote_raw)
        ctypes.memmove(local, source, size)


@pytest.mark.parametrize(("local_tp", "remote_tp"), [(8, 8), (4, 8), (8, 4), (2, 8), (8, 2)])
@pytest.mark.parametrize("replicated", [False, True])
def test_qwen_state_payload_and_padding(local_tp, remote_tp, replicated):
    # Scale widths down but preserve Q:K:V = 16:16:48 and both GDN state axes.
    def make_rank(tp, rank, padding):
        if replicated:
            shapes = [(3, 40)]
        else:
            shapes = [(3, 80 // tp), (48 // tp, 1, 2)]
        pairs = [padded_tensor((4, *s), torch.float32, padding) for s in shapes]
        for index, (_, tensor) in enumerate(pairs):
            for block in range(4):
                if replicated:
                    tensor[block] = torch.arange(120).reshape(3, 40) + block * 1000
                elif index == 0:
                    global_projections = [
                        torch.arange(n).reshape(1, n).expand(3, n) + j * 100 for j, n in enumerate((16, 16, 48))
                    ]
                    tensor[block] = (
                        torch.cat(
                            [
                                p[:, rank * p.shape[1] // tp : (rank + 1) * p.shape[1] // tp]
                                for p in global_projections
                            ],
                            dim=1,
                        )
                        + block * 1000
                    )
                else:
                    tensor[block] = (
                        torch.arange(96).reshape(48, 1, 2)[rank * 48 // tp : (rank + 1) * 48 // tp] + block * 1000
                    )
        return pairs

    producers = {r: make_rank(remote_tp, r, 32) for r in range(remote_tp)}
    for local_rank in range(local_tp):
        pairs = make_rank(local_tp, local_rank, 64)
        raw, caches = zip(*pairs)
        expected = [t.clone() for t in caches]
        for tensor in caches:
            tensor.zero_()
        spec = MambaSpec(
            block_size=16,
            shapes=tuple(tuple(t.shape[1:]) for t in caches),
            dtypes=(torch.float32,) * len(caches),
            tp_replicated=replicated,
            mamba_type=MambaAttentionBackendEnum.SHORT_CONV if replicated else MambaAttentionBackendEnum.GDN_ATTN,
        )
        thread = make_thread(
            tp_size=local_tp,
            tp_rank=local_rank,
            kv_cache_specs=[spec],
            kv_caches_base_addr=[[t.data_ptr() for t in caches]],
            **cache_fields(caches),
        )
        remote = make_pp_metadata(
            **cache_fields([p[1] for p in producers[0]]),
            tp_base_addrs={r: [[t.data_ptr() for _, t in ps]] for r, ps in producers.items()},
        )
        groups = thread._get_layer_remote_tp_rank_groups(0, 0, spec, remote, 1, remote_tp, 1)
        if replicated:
            assert groups == [list(range(remote_tp))]
        for group in groups:
            rank = group[-1]
            execute_read(thread, remote, remote_tp, rank, raw, [p[0] for p in producers[rank]])
        for tensor, reference, backing in zip(caches, expected, raw):
            assert torch.equal(tensor[2], reference[1])
            assert torch.equal(tensor[0], reference[3])
            assert torch.count_nonzero(tensor[1]) == torch.count_nonzero(tensor[3]) == 0
            stride = tensor.stride(0) * tensor.element_size()
            payload = math.prod(tensor.shape[1:]) * tensor.element_size()
            assert torch.all(backing.reshape(4, stride)[:, payload:] == 0xA5)


def test_nhd_registration_scheduler_and_wire_roundtrip(monkeypatch):
    _, key = padded_tensor((4, 8, 1, 4), torch.float32, 64)
    _, value = padded_tensor((4, 8, 1, 4), torch.float32, 64)
    spec = make_full_spec(block_size=8)
    worker = base_worker.MooncakeBaseConnectorWorker.__new__(base_worker.MooncakeBaseConnectorWorker)
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 8
    worker.engine_id, worker.te_rpc_port, worker.block_size = "engine-p", 9000, 8
    worker.side_channel_host, worker.handshake_port = "127.0.0.1", 5000
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[make_kv_cache_tensor(size=2048, layers=["qsa"], layer_stride=2048, block_stride=512)],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["qsa"], kv_cache_spec=spec)],
    )
    monkeypatch.setattr(base_worker, "global_te", MagicMock())
    worker.register_kv_caches({"qsa": (key, value)})
    metadata = worker.transfer_metadata
    assert metadata.block_layouts == ["NHD"]
    assert metadata.block_shapes == [[(8, 1, 4), (8, 1, 4)]]
    sender = make_sending_thread({rank: metadata for rank in range(8)}, tp_size=8)
    decoded = msgspec.msgpack.decode(sender.encoded_metadata, type=MooncakeTransferMetadataGroups)
    assert decoded.metadata_by_pp_rank[0].block_layouts == ["NHD"]
    with pytest.raises(ValueError, match="block_layouts"):
        make_sending_thread({0: metadata, 1: replace(metadata, block_layouts=["HND"])}, tp_size=2)


@pytest.mark.parametrize("layouts", [[], ["XYZ"], ["HND", "NHD"]])
def test_invalid_layout_metadata(layouts):
    with pytest.raises(ValueError, match="block_layouts"):
        replace(make_transfer_metadata(), block_layouts=layouts)


@pytest.mark.parametrize(("local_tp", "remote_tp"), [(8, 8), (4, 8), (8, 4), (2, 8), (8, 2), (1, 8), (8, 1)])
@pytest.mark.parametrize(("local_layout", "remote_layout"), [("NHD", "NHD"), ("NHD", "HND"), ("HND", "NHD")])
def test_qsa_attention_payload_and_padding(local_tp, remote_tp, local_layout, remote_layout):
    def make_rank(tp, rank, layout, padding):
        count = max(1, 2 // tp)
        first_head = rank * count if tp <= 2 else rank // (tp // 2)
        shape = (4, 8, count, 4) if layout == "NHD" else (4, count, 8, 4)
        pairs = [padded_tensor(shape, torch.float32, padding) for _ in range(2)]
        for component, (_, tensor) in enumerate(pairs):
            logical = tensor if layout == "NHD" else tensor.transpose(1, 2)
            global_values = torch.arange(4 * 8 * 2 * 4).reshape(4, 8, 2, 4) + component * 1000
            logical.copy_(global_values[:, :, first_head : first_head + count])
        return pairs

    producers = {r: make_rank(remote_tp, r, remote_layout, 32) for r in range(remote_tp)}
    for local_rank in range(local_tp):
        pairs = make_rank(local_tp, local_rank, local_layout, 64)
        raw, caches = zip(*pairs)
        references = [t.clone() for t in caches]
        for tensor in caches:
            tensor.zero_()
        spec = make_full_spec(block_size=8, num_kv_heads=max(1, 2 // local_tp))
        fields = cache_fields(caches)
        metadata = replace(
            make_transfer_metadata(
                **fields,
                base_addrs=[[t.data_ptr() for t in caches]],
                block_size_scales=[[1, 1]],
            ),
            block_layouts=[local_layout],
        )
        thread = make_thread(
            tp_size=local_tp,
            tp_rank=local_rank,
            num_key_value_heads=2,
            kv_cache_specs=[spec],
            local_metadata=metadata,
            kv_caches_base_addr=[[t.data_ptr() for t in caches]],
            **fields,
        )
        remote = replace(
            make_pp_metadata(
                **cache_fields([p[1] for p in producers[0]]),
                tp_base_addrs={r: [[t.data_ptr() for _, t in ps]] for r, ps in producers.items()},
            ),
            block_layouts=[remote_layout],
        )
        groups = thread._get_layer_remote_tp_rank_groups(0, 0, spec, remote, 1, remote_tp, 1)
        for group in groups:
            rank = group[-1]
            execute_read(thread, remote, remote_tp, rank, raw, [p[0] for p in producers[rank]])
        for tensor, reference, backing in zip(caches, references, raw):
            assert torch.equal(tensor[2], reference[1])
            assert torch.equal(tensor[0], reference[3])
            assert torch.count_nonzero(tensor[1]) == torch.count_nonzero(tensor[3]) == 0
            stride = tensor.stride(0) * tensor.element_size()
            payload = math.prod(tensor.shape[1:]) * tensor.element_size()
            assert torch.all(backing.reshape(4, stride)[:, payload:] == 0xA5)
