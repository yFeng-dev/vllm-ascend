# SPDX-License-Identifier: Apache-2.0
import math
import unittest
from types import SimpleNamespace

import torch
from vllm_ascend.worker.qsa_private_cache import make_private_qsa_views


class PageViewRunner:
    @staticmethod
    def _page_strided_tensor_view(raw, *, dtype, shape, block_stride_bytes, storage_offset_bytes):
        itemsize = torch.empty((), dtype=dtype).element_size()
        strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
        strides[0] = block_stride_bytes // itemsize
        return torch.as_strided(raw.view(dtype), shape, strides, storage_offset_bytes // itemsize)


class TestPrivateQSACache(unittest.TestCase):
    def test_dense_kv_rows_and_disjoint_writes(self):
        for heads in (1, 2):
            with self.subTest(heads=heads):
                spec = SimpleNamespace(
                    dtype=torch.bfloat16, block_size=8, num_kv_heads=heads, head_size=16, page_size_bytes=2048
                )
                raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
                key, value = make_private_qsa_views(
                    PageViewRunner(), raw, spec, 3, SimpleNamespace(uses_qsa_page_major_kv_view=True)
                )
                self.assertEqual(key.shape, (3, 8, heads, 16))
                self.assertEqual(key.stride(1), heads * 16)
                self.assertEqual(value.stride(1), heads * 16)
                self.assertEqual(value.data_ptr() - key.data_ptr(), 8 * heads * 16 * 2)
                key.fill_(1)
                value.fill_(2)
                self.assertTrue(torch.all(key == 1))
                self.assertTrue(torch.all(value == 2))
                self.assertEqual(key.untyped_storage().data_ptr(), raw.untyped_storage().data_ptr())

    def test_page_padding_and_neighboring_blocks_are_preserved(self):
        spec = SimpleNamespace(dtype=torch.bfloat16, block_size=8, num_kv_heads=1, head_size=16, page_size_bytes=2048)
        raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
        key, value = make_private_qsa_views(
            PageViewRunner(), raw, spec, 3, SimpleNamespace(uses_qsa_page_major_kv_view=True)
        )
        key[1].fill_(3)
        value[1].fill_(4)
        self.assertEqual(key.stride(0) * key.element_size(), 2048)
        self.assertTrue(torch.all(raw[:2048] == 0))
        self.assertTrue(torch.all(raw[2048 + 512 :] == 0))

    def test_nonzero_storage_offset(self):
        backing = torch.zeros(128 + 3 * 2048, dtype=torch.int8)
        backing[:128].fill_(7)
        raw = backing[128:]
        spec = SimpleNamespace(dtype=torch.bfloat16, block_size=8, num_kv_heads=1, head_size=16, page_size_bytes=2048)
        key, value = make_private_qsa_views(
            PageViewRunner(), raw, spec, 3, SimpleNamespace(uses_qsa_page_major_kv_view=True)
        )
        self.assertEqual(key.data_ptr(), raw.data_ptr())
        key.fill_(1)
        value.fill_(2)
        self.assertTrue(torch.all(backing[:128] == 7))

    def test_state_views_use_spec_dimensions(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                itemsize = torch.empty((), dtype=dtype).element_size()
                spec = SimpleNamespace(
                    dtype=dtype, num_heads=1, num_states=4, state_content_size_bytes=32, page_size_bytes=512
                )
                raw = torch.zeros(3 * 512, dtype=torch.int8)
                state = make_private_qsa_views(PageViewRunner(), raw, spec, 3, SimpleNamespace())
                self.assertEqual(state.shape, (3, 1, 4, 32 // itemsize))
                self.assertEqual(state.stride(0) * itemsize, 512)
                state[1].fill_(1)
                self.assertTrue(torch.all(raw[:512] == 0))
                self.assertTrue(torch.all(raw[512 + 128 :] == 0))

    def test_rejects_page_too_small_for_both_components(self):
        spec = SimpleNamespace(dtype=torch.bfloat16, block_size=8, num_kv_heads=1, head_size=16, page_size_bytes=256)
        with self.assertRaises(AssertionError):
            make_private_qsa_views(
                PageViewRunner(),
                torch.zeros(3 * 256, dtype=torch.int8),
                spec,
                3,
                SimpleNamespace(uses_qsa_page_major_kv_view=True),
            )


if __name__ == "__main__":
    unittest.main()
