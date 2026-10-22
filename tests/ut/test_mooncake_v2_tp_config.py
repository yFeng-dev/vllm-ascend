# SPDX-License-Identifier: Apache-2.0
import pytest
from tests.ut.test_ascend_config import VllmConfig
from vllm.config import DeviceConfig, KVTransferConfig
from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig


@pytest.mark.parametrize("p_tp,d_tp", [(4, 8), (8, 4), (8, 8)])
@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakeConnector"])
def test_pull_topology_does_not_enable_legacy_head_rearrangement(p_tp, d_tp, connector):
    vc = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vc.model_config.get_total_num_kv_heads = lambda: 2
    vc.kv_transfer_config = KVTransferConfig(
        kv_connector=connector,
        kv_role="kv_producer",
        kv_connector_extra_config={"prefill": {"tp_size": p_tp}, "decode": {"tp_size": d_tp}},
    )
    config = AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig())
    if connector == "MooncakeConnector" and p_tp < d_tp:
        with pytest.raises(AssertionError, match="Prefill TP size must"):
            config.derive_and_validate(vc)
    else:
        config.derive_and_validate(vc)
        assert config.pd_tp_ratio == (1 if connector == "MooncakeConnectorV2" else p_tp // d_tp)
        assert config.pd_head_ratio == 1
