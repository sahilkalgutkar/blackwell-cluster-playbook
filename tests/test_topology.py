import json

import pytest

from bcp.topology import analyze, identify, render_json, render_mermaid, render_text


def load(read, profile):
    def opt(key):
        try:
            return read(profile, key)
        except FileNotFoundError:
            return None

    return analyze(opt("gpu_query"), opt("topo"), opt("smi_q"), opt("dgx_release"))


@pytest.mark.parametrize(
    "name, model, generation, superchip",
    [
        ("NVIDIA GB200", "GB200", "Blackwell", True),
        ("NVIDIA B200", "B200", "Blackwell", False),
        ("NVIDIA GB300", "GB300", "Blackwell Ultra", True),
        ("NVIDIA B300 SXM6 AC", "B300", "Blackwell Ultra", False),
        ("NVIDIA H100 80GB HBM3", "H100", "Hopper", False),
        ("NVIDIA GH200 480GB", "GH200", "Hopper", True),
        ("NVIDIA H200", "H200", "Hopper", False),
        ("NVIDIA GB10", "GB10", "Blackwell", True),
    ],
)
def test_identify(name, model, generation, superchip):
    assert identify(name) == (model, generation, superchip)


def test_identify_unknown_falls_back_to_architecture():
    assert identify("NVIDIA Future GPU", "Feynman") == ("unknown", "Feynman", False)


def test_hgx_b200_is_dgx(read):
    t = load(read, "dgx-b200")
    p = t.platform
    assert p.form_factor == "HGX 8-GPU baseboard (NVSwitch)"
    assert p.dgx == "DGX Server for DGX B200"
    assert p.nvlink_links == 18 and p.nvlink_symmetric
    assert p.nominal_nvlink_gbps == 1800
    assert not p.multi_node_nvlink
    assert p.warnings == []
    assert t.nearest_nic["GPU3"] == ["NIC3 (mlx5_3, PIX)"]


def test_hgx_h100(read):
    p = load(read, "hgx-h100").platform
    assert p.generation == "Hopper"
    assert p.dgx is None
    assert p.nominal_nvlink_gbps == 900


def test_nvl72_tray(read):
    p = load(read, "gb200-nvl72").platform
    assert p.multi_node_nvlink
    assert p.form_factor.startswith("NVL72-class rack node")
    assert p.clique_id == "32766"
    assert p.gpu_count == 4


def test_degraded_gpu_is_singled_out(read):
    t = load(read, "dgx-b200-degraded")
    assert t.weak_nvlink_gpus == {"GPU5": 16}
    assert t.platform.nvlink_symmetric is False
    assert any("GPU5" in w for w in t.platform.warnings)


def test_superchip_without_fabric_warns():
    smi_q = (
        "Driver Version : 580.95.05\n"
        "GPU 00000000:01:00.0\n"
        "    Product Name : NVIDIA GB200\n"
        "    Fabric\n"
        "        State : In Progress\n"
        "        Status : N/A\n"
    )
    p = analyze("0, NVIDIA GB200, 00000000:01:00.0, 189471 MiB, 580.95.05\n", None, smi_q).platform
    assert not p.multi_node_nvlink
    assert any("fabric state" in w for w in p.warnings)


def test_renderers(read):
    t = load(read, "dgx-b200-degraded")
    text = render_text(t)
    assert "ASYMMETRIC" in text and "GPU5" in text
    mermaid = render_mermaid(t)
    assert mermaid.startswith("flowchart LR")
    assert "GPU5 ===|NV16| FABRIC" in mermaid
    assert "class GPU5 warn" in mermaid
    doc = json.loads(render_json(t))
    assert doc["platform"]["gpu_model"] == "B200"
    assert doc["links"]["GPU0-GPU5"] == "NV16"


def test_nvl72_mermaid_mentions_rack_fabric(read):
    assert "NVLink Switch trays" in render_mermaid(load(read, "gb200-nvl72"))
