from bcp.parsers import (
    link_rank,
    nvlink_count,
    parse_dcgm,
    parse_dgx_release,
    parse_gpu_query,
    parse_nccl,
    parse_nvbandwidth,
    parse_smi_q,
    parse_topo,
)


def test_gpu_query(read):
    gpus = parse_gpu_query(read("dgx-b200", "gpu_query"))
    assert len(gpus) == 8
    assert gpus[0].name == "NVIDIA B200"
    assert gpus[0].memory_mib == 183359
    assert gpus[7].index == 7


def test_topo_matrix(read):
    topo = parse_topo(read("dgx-b200", "topo"))
    assert topo.gpus == [f"GPU{i}" for i in range(8)]
    assert len(topo.nics) == 10
    assert topo.link("GPU0", "GPU1") == "NV18"
    assert topo.link("GPU0", "NIC0") == "PIX"
    assert topo.link("GPU0", "NIC4") == "SYS"
    assert topo.cpu_affinity["GPU0"] == "0-55,112-167"
    assert topo.numa_affinity["GPU7"] == "1"
    assert topo.nic_names["NIC9"] == "mlx5_9"


def test_topo_handles_ansi_and_space_separated_output():
    text = (
        "\x1b[4m        GPU0    GPU1    NIC0    CPU Affinity    NUMA Affinity\x1b[0m\n"
        "GPU0     X      NV18    PIX     0-31    0\n"
        "GPU1    NV18     X      NODE    0-31    0\n"
        "NIC0    PIX     NODE     X\n\n"
        "NIC Legend:\n\n  NIC0: mlx5_0\n"
    )
    topo = parse_topo(text)
    assert topo.link("GPU1", "GPU0") == "NV18"
    assert topo.link("GPU1", "NIC0") == "NODE"
    assert topo.cpu_affinity["GPU1"] == "0-31"
    assert topo.nic_names == {"NIC0": "mlx5_0"}


def test_link_helpers():
    assert nvlink_count("NV18") == 18
    assert nvlink_count("PIX") is None
    assert link_rank("NV4") < link_rank("PIX") < link_rank("NODE") < link_rank("SYS")


def test_smi_q_fabric(read):
    q = parse_smi_q(read("gb200-nvl72", "smi_q"))
    assert q.driver == "580.95.05"
    assert q.cuda == "13.0"
    assert len(q.gpus) == 4
    assert q.gpus[0].architecture == "Blackwell"
    assert q.gpus[0].clique_id == "32766"
    assert all(g.in_multi_node_domain for g in q.gpus)


def test_smi_q_single_node_fabric_is_not_multi_node(read):
    q = parse_smi_q(read("dgx-b200", "smi_q"))
    assert q.gpus[0].state == "Completed"
    assert not q.gpus[0].in_multi_node_domain  # all-zero ClusterUUID
    h100 = parse_smi_q(read("hgx-h100", "smi_q"))
    assert not any(g.in_multi_node_domain for g in h100.gpus)


def test_dgx_release(read):
    info = parse_dgx_release(read("dgx-b200", "dgx_release"))
    assert info["DGX_PLATFORM"] == "DGX Server for DGX B200"


def test_nccl_all_reduce(read):
    r = parse_nccl(read("dgx-b200", "nccl_all_reduce"))
    assert r.nccl_version == "2.32.6"
    assert r.nranks == 8
    assert r.out_of_bounds_ok is True
    assert r.rows[0].size == 8 and r.rows[-1].size == 8 * 1024**3
    assert r.peak.best_busbw > 800
    assert r.wrong_total == 0
    assert r.avg_busbw is not None


def test_nccl_alltoall_in_place_is_na(read):
    r = parse_nccl(read("dgx-b200", "nccl_alltoall"))
    assert all(row.ip_busbw is None for row in r.rows)
    assert r.peak.best_busbw == max(row.oop_busbw for row in r.rows)


def test_nccl_counts_wrong_results():
    text = (
        "#  Rank  0 Group  0 Pid 1 on n device  0 [0x1b] NVIDIA B200\n"
        "     1048576        262144     float     sum      -1    20.00   52.43   91.75      3"
        "    20.00   52.43   91.75      0\n"
    )
    r = parse_nccl(text)
    assert r.wrong_total == 3
    assert r.rows[0].root == -1


def test_nccl_root_zero_is_kept():
    text = "     1048576        262144     float     sum       0    20.00   52.43   91.75      0\n"
    assert parse_nccl(text).rows[0].root == 0


def test_nvbandwidth(read):
    r = parse_nvbandwidth(read("dgx-b200", "nvbandwidth"))
    assert r.version == "v0.11.0"
    d2d = r.tests["device_to_device_memcpy_read_ce"]
    assert len(d2d.cells) == 8 * 7  # diagonal is N/A
    assert abs(d2d.avg_gbps - 771.4) < 1
    assert abs(d2d.sum_gbps - sum(v for _, _, v in d2d.cells)) < 0.1
    h2d = r.tests["host_to_device_memcpy_ce"]
    assert len(h2d.cells) == 8


def test_nvbandwidth_finds_weak_pair(read):
    d2d = parse_nvbandwidth(read("dgx-b200-degraded", "nvbandwidth")).tests[
        "device_to_device_memcpy_read_ce"
    ]
    row, col, _ = d2d.weakest
    assert 5 in (row, col)


def test_dcgm_pass_and_fail(read):
    ok = parse_dcgm(read("dgx-b200", "dcgm_diag"))
    assert ok.overall == "pass"
    assert ok.version == "4.7.0"
    assert {"software", "pcie", "memory", "targeted_stress"} <= set(ok.tests)

    bad = parse_dcgm(read("dgx-b200-degraded", "dcgm_diag"))
    assert bad.overall == "fail"
    assert bad.tests["pcie"] == "fail"
    assert any("NVLink" in m for m in bad.messages)


def test_dcgm_tolerates_other_shapes_and_prefixes():
    text = 'Some banner line\n{"tests": [{"name": "memory", "results": [{"status": "WARN"}]}]}'
    r = parse_dcgm(text)
    assert r.tests == {"memory": "warn"}
    assert parse_dcgm("not json at all").error
