#!/usr/bin/env python3
"""Generate the synthetic fixtures in fixtures/.

These files imitate real tool output closely enough to exercise every parser and check, so CI
runs without GPUs. The numbers are modelled, NOT measured: every manifest says
"synthetic": true and reports label such runs. Replace them with `bcp collect` output from real
hardware as soon as you have it.

    python scripts/make_fixtures.py            # write fixtures/
    python scripts/make_fixtures.py --check    # exit 1 if fixtures/ is stale
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NCCL_VERSION = "2.32.6+cuda13.0"
DRIVER = "580.95.05"
CUDA = "13.0"


@dataclass
class Profile:
    name: str
    host: str
    gpu_name: str
    architecture: str
    gpus: int
    memory_mib: int
    nvlinks: int
    numa_split: int  # GPUs per NUMA node
    cpu_affinity: tuple[str, ...]
    nics: list[tuple[str, int | None]]  # (device name, GPU it sits next to or None)
    # Peak bus bandwidth (GB/s) the fixture curve converges to, per collective.
    busbw: dict[str, float]
    d2d_gbps: float
    h2d_gbps: float
    fabric: dict[str, str] | None = None
    dgx_release: str | None = None
    degraded_gpu: int | None = None
    degraded_links: int | None = None
    dcgm_fail: dict[str, str] = field(default_factory=dict)  # test -> message


PROFILES = [
    Profile(
        name="hgx-h100",
        host="synthetic-hgx-h100",
        gpu_name="NVIDIA H100 80GB HBM3",
        architecture="Hopper",
        gpus=8,
        memory_mib=81559,
        nvlinks=18,
        numa_split=4,
        cpu_affinity=("0-55,112-167", "56-111,168-223"),
        nics=[(f"mlx5_{i}", i) for i in range(8)],
        busbw={
            "all_reduce": 480.0,
            "all_gather": 365.0,
            "reduce_scatter": 362.0,
            "alltoall": 340.0,
        },
        d2d_gbps=389.7,
        h2d_gbps=55.6,
        fabric={"State": "N/A", "Status": "N/A", "CliqueId": "N/A", "ClusterUUID": "N/A"},
    ),
    Profile(
        name="dgx-b200",
        host="synthetic-dgx-b200",
        gpu_name="NVIDIA B200",
        architecture="Blackwell",
        gpus=8,
        memory_mib=183359,
        nvlinks=18,
        numa_split=4,
        cpu_affinity=("0-55,112-167", "56-111,168-223"),
        nics=[(f"mlx5_{i}", i) for i in range(8)] + [("mlx5_8", None), ("mlx5_9", None)],
        busbw={
            "all_reduce": 820.0,
            "all_gather": 690.0,
            "reduce_scatter": 685.0,
            "alltoall": 640.0,
        },
        d2d_gbps=771.4,
        h2d_gbps=55.9,
        fabric={
            "State": "Completed",
            "Status": "Success",
            "CliqueId": "0",
            "ClusterUUID": "00000000-0000-0000-0000-000000000000",
        },
        dgx_release=(
            'DGX_NAME="DGX Server"\n'
            'DGX_PRETTY_NAME="NVIDIA DGX Server"\n'
            'DGX_SWBUILD_VERSION="7.2.0"\n'
            'DGX_PLATFORM="DGX Server for DGX B200"\n'
        ),
    ),
    Profile(
        name="dgx-b200-degraded",
        host="synthetic-dgx-b200-degraded",
        gpu_name="NVIDIA B200",
        architecture="Blackwell",
        gpus=8,
        memory_mib=183359,
        nvlinks=18,
        numa_split=4,
        cpu_affinity=("0-55,112-167", "56-111,168-223"),
        nics=[(f"mlx5_{i}", i) for i in range(8)] + [("mlx5_8", None), ("mlx5_9", None)],
        # A GPU with 2 of 18 NVLinks down drags every collective down to its speed.
        busbw={
            "all_reduce": 560.0,
            "all_gather": 610.0,
            "reduce_scatter": 605.0,
            "alltoall": 470.0,
        },
        d2d_gbps=771.4,
        h2d_gbps=55.9,
        fabric={
            "State": "Completed",
            "Status": "Success",
            "CliqueId": "0",
            "ClusterUUID": "00000000-0000-0000-0000-000000000000",
        },
        degraded_gpu=5,
        degraded_links=16,
        dcgm_fail={"pcie": "GPU 5: NVLink 7 and 13 are down (link state inactive)"},
    ),
    Profile(
        name="gb200-nvl72",
        host="synthetic-gb200-nvl72-tray07",
        gpu_name="NVIDIA GB200",
        architecture="Blackwell",
        gpus=4,
        memory_mib=189471,
        nvlinks=18,
        numa_split=2,
        cpu_affinity=("0-71", "72-143"),
        nics=[
            ("mlx5_0", 0),
            ("mlx5_1", 1),
            ("mlx5_2", 2),
            ("mlx5_3", 3),
            ("mlx5_4", None),
            ("mlx5_5", None),
        ],
        busbw={
            "all_reduce": 700.0,
            "all_gather": 610.0,
            "reduce_scatter": 600.0,
            "alltoall": 560.0,
        },
        d2d_gbps=768.2,
        h2d_gbps=355.0,  # Grace -> GPU over NVLink-C2C, not PCIe
        fabric={
            "State": "Completed",
            "Status": "Success",
            "CliqueId": "32766",
            "ClusterUUID": "7f3c2a9e-4b1d-4e8a-9c6f-2d5b8e1a0c34",
        },
    ),
]


def numa(p: Profile, gpu: int) -> int:
    return gpu // p.numa_split


def bus_id(gpu: int) -> str:
    return f"00000000:{0x18 + gpu * 0x10:02X}:00.0"


def gpu_query(p: Profile) -> str:
    return "".join(
        f"{i}, {p.gpu_name}, {bus_id(i)}, {p.memory_mib} MiB, {DRIVER}\n" for i in range(p.gpus)
    )


def link(p: Profile, a: int, b: int) -> str:
    if a == b:
        return "X"
    if p.degraded_gpu in (a, b):
        return f"NV{p.degraded_links}"
    return f"NV{p.nvlinks}"


def nic_link(p: Profile, gpu: int, nic_gpu: int | None, nic_index: int) -> str:
    if nic_gpu == gpu:
        return "PIX"
    home = numa(p, nic_gpu) if nic_gpu is not None else nic_index % 2
    return "NODE" if home == numa(p, gpu) else "SYS"


def topo(p: Profile) -> str:
    gpus = [f"GPU{i}" for i in range(p.gpus)]
    nics = [f"NIC{i}" for i in range(len(p.nics))]
    header = "\t" + "\t".join(gpus + nics) + "\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\n"
    rows = []
    for g in range(p.gpus):
        cells = [link(p, g, o) for o in range(p.gpus)]
        cells += [nic_link(p, g, ng, i) for i, (_, ng) in enumerate(p.nics)]
        gpu_numa = str(2 + 8 * g) if p.numa_split == 2 else "N/A"
        aff = p.cpu_affinity[numa(p, g)]
        rows.append(f"GPU{g}\t " + "\t".join(cells) + f"\t{aff}\t{numa(p, g)}\t\t{gpu_numa}\n")
    for i, (_, ng) in enumerate(p.nics):
        cells = [nic_link(p, g, ng, i) for g in range(p.gpus)]
        cells += [
            "X"
            if j == i
            else (
                "PIX"
                if ng is not None and ng == nj
                else (
                    "NODE"
                    if (numa(p, ng) if ng is not None else i % 2)
                    == (numa(p, nj) if nj is not None else j % 2)
                    else "SYS"
                )
            )
            for j, (_, nj) in enumerate(p.nics)
        ]
        rows.append(f"NIC{i}\t " + "\t".join(cells) + "\n")
    legend = (
        "\nLegend:\n\n"
        "  X    = Self\n"
        "  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes"
        " (e.g., QPI/UPI)\n"
        "  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges"
        " within a NUMA node\n"
        "  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)\n"
        "  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host"
        " Bridge)\n"
        "  PIX  = Connection traversing at most a single PCIe bridge\n"
        "  NV#  = Connection traversing a bonded set of # NVLinks\n\n"
        "NIC Legend:\n\n"
    )
    legend += "".join(f"  NIC{i}: {name}\n" for i, (name, _) in enumerate(p.nics))
    return header + "".join(rows) + legend


def smi_q(p: Profile) -> str:
    out = [
        "",
        "==============NVSMI LOG==============",
        "",
        "Timestamp                                 : Thu Nov 12 09:14:02 2026",
        f"Driver Version                            : {DRIVER}",
        f"CUDA Version                              : {CUDA}",
        "",
        f"Attached GPUs                             : {p.gpus}",
    ]
    for g in range(p.gpus):
        out += [
            f"GPU {bus_id(g)}",
            f"    Product Name                          : {p.gpu_name}",
            "    Product Brand                         : NVIDIA",
            f"    Product Architecture                  : {p.architecture}",
            "    Persistence Mode                      : Enabled",
        ]
        if p.fabric:
            out.append("    Fabric")
            for key in ("State", "Status", "CliqueId", "ClusterUUID"):
                out.append(f"        {key:<34}: {p.fabric[key]}")
            if p.fabric.get("State") == "Completed":
                out += ["        Health", "            Bandwidth                     : Full"]
        out.append("    Processes                             : None")
    return "\n".join(out) + "\n"


FACTORS = {  # busbw = algbw * factor(n)
    "all_reduce": lambda n: 2 * (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
    "alltoall": lambda n: (n - 1) / n,
}
BINARIES = {
    "all_reduce": "all_reduce_perf",
    "all_gather": "all_gather_perf",
    "reduce_scatter": "reduce_scatter_perf",
    "alltoall": "alltoall_perf",
}


def nccl(p: Profile, op: str) -> str:
    n = p.gpus
    factor = FACTORS[op](n)
    peak_algbw = p.busbw[op] / factor  # GB/s
    latency_us = {"all_reduce": 14.0, "all_gather": 10.0, "reduce_scatter": 10.5, "alltoall": 16.0}[
        op
    ]
    max_bytes = 8 * 1024**3
    lines = [
        f"# Collective test starting: {BINARIES[op]}",
        f"# nThread 1 nGpus {n} minBytes 8 maxBytes {max_bytes} step: 2(factor) warmup iters: 1 "
        "iters: 20 agg iters: 1 validation: 1 graph: 0",
        "#",
        "# Using devices",
    ]
    for g in range(n):
        lines.append(
            f"#  Rank {g:2d} Group  0 Pid 412233 on {p.host} device {g:2d} "
            f"[0000:{0x18 + g * 0x10:02x}:00] {p.gpu_name}"
        )
    lines += [
        f"NCCL version {NCCL_VERSION}",
        "#",
        "#                                                              out-of-place"
        "                       in-place          ",
        "#       size         count      type   redop    root     time   algbw   busbw #wrong"
        "     time   algbw   busbw #wrong",
        "#        (B)    (elements)                               (us)  (GB/s)  (GB/s)       "
        "     (us)  (GB/s)  (GB/s)       ",
    ]
    redop = "sum" if op in ("all_reduce", "reduce_scatter") else "none"
    busbws = []
    size = 8
    while size <= max_bytes:
        # all_gather/reduce_scatter/alltoall need a multiple of n elements.
        elems = size // 4
        if op != "all_reduce":
            elems = (elems // n) * n
        eff = elems * 4
        count = elems if op == "all_reduce" else elems // n
        if eff == 0:
            t = 0.2
            algbw = busbw = 0.0
        else:
            t = latency_us + eff / (peak_algbw * 1e3)  # GB/s -> bytes/us is *1e3
            algbw = eff / t / 1e3
            busbw = algbw * factor
        busbws.append(busbw)
        oop = f"{t:8.2f} {algbw:7.2f} {busbw:7.2f} {0:6d}"
        ip = (
            f"{t * 0.99:8.2f} {algbw * 1.01:7.2f} {busbw * 1.01:7.2f} {0:6d}"
            if op != "alltoall"
            else f"{'N/A':>8} {'N/A':>7} {'N/A':>7} {'N/A':>6}"
        )
        lines.append(f"{eff:12d} {count:13d} {'float':>9} {redop:>7} {-1:7d} {oop} {ip}")
        size *= 2
    avg = sum(busbws) / len(busbws)
    lines += [
        "# Out of bounds values : 0 OK",
        f"# Avg bus bandwidth    : {avg:.4f} ",
        "#",
        f"# Collective test concluded: {BINARIES[op]}",
        "",
    ]
    return "\n".join(lines)


def nvbandwidth(p: Profile) -> str:
    n = p.gpus
    out = [
        "nvbandwidth Version: v0.11.0",
        "Built from Git version: v0.11.0",
        "",
        f"CUDA Runtime Version: {CUDA.replace('.', '0')}0",
        f"CUDA Driver Version: {CUDA.replace('.', '0')}0",
        f"Driver Version: {DRIVER}",
        "",
    ]
    out += [f"Device {g}: {p.gpu_name} ({bus_id(g)[4:]})" for g in range(n)]
    out.append("")

    def matrix(name: str, title: str, rows: list[list[float | None]], cols: int) -> None:
        out.append(f"Running {name}.")
        out.append(title)
        out.append("     " + "".join(f"{c:>10d}" for c in range(cols)))
        total = 0.0
        for r, row in enumerate(rows):
            cells = "".join(f"{'N/A':>10}" if v is None else f"{v:>10.2f}" for v in row)
            out.append(f"{r:>3d}  {cells}")
            total += sum(v for v in row if v is not None)
        out.append("")
        out.append(f"SUM {name} {total:.2f}")
        out.append("")

    def d2d(a: int, b: int) -> float | None:
        if a == b:
            return None
        value = p.d2d_gbps + ((a * 7 + b * 3) % 5 - 2) * 0.11
        if p.degraded_gpu in (a, b):
            value *= p.degraded_links / p.nvlinks
        return round(value, 2)

    matrix(
        "device_to_device_memcpy_read_ce",
        "memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)",
        [[d2d(a, b) for b in range(n)] for a in range(n)],
        n,
    )
    h2d = [[round(p.h2d_gbps + (g % 3) * 0.07, 2) for g in range(n)]]
    matrix("host_to_device_memcpy_ce", "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)", h2d, n)
    d2h = [[round(p.h2d_gbps * 0.98 + (g % 2) * 0.05, 2) for g in range(n)]]
    matrix("device_to_host_memcpy_ce", "memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)", d2h, n)
    return "\n".join(out)


DCGM_TESTS = [
    ("Deployment", ["software"]),
    ("Hardware", ["memory", "pcie", "diagnostic"]),
    ("Integration", ["memory_bandwidth"]),
    ("Stress", ["targeted_stress", "targeted_power"]),
]


def dcgm(p: Profile) -> str:
    categories = []
    for category, names in DCGM_TESTS:
        tests = []
        for name in names:
            if name == "software":
                results = [{"status": "Pass"}]
            else:
                results = []
                for g in range(p.gpus):
                    entry = {
                        "entity_group": "GPU",
                        "entity_id": g,
                        "gpu_id": str(g),
                        "status": "Pass",
                    }
                    if name in p.dcgm_fail and g == p.degraded_gpu:
                        entry["status"] = "Fail"
                        entry["warnings"] = [{"warning": p.dcgm_fail[name], "error_id": 66}]
                    results.append(entry)
            tests.append({"name": name, "results": results})
        categories.append({"category": category, "tests": tests})
    doc = {
        "version": "4.7.0",
        "DCGM Diagnostic": {"test_categories": categories},
        "driver_version_detected": DRIVER,
    }
    return json.dumps(doc, indent=2) + "\n"


def write_profile(p: Profile, base: Path) -> None:
    d = base / p.name
    d.mkdir(parents=True, exist_ok=True)
    files = {
        "gpu_query": gpu_query(p),
        "topo": topo(p),
        "smi_q": smi_q(p),
        "dcgm_diag": dcgm(p),
        "nvbandwidth": nvbandwidth(p),
        **{f"nccl_{op}": nccl(p, op) for op in BINARIES},
    }
    if p.dgx_release:
        files["dgx_release"] = p.dgx_release
    for key, text in files.items():
        (d / f"{key}.txt").write_text(text)
    manifest = {
        "schema": 1,
        "host": p.host,
        "collected_at": "2026-11-12T09:14:02Z",
        "synthetic": True,
        "note": "Generated by scripts/make_fixtures.py. Modelled numbers, not a measurement.",
        "probes": {
            key: {"exit_code": 1 if key == "dcgm_diag" and p.dcgm_fail else 0} for key in files
        },
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def generate(base: Path) -> None:
    for profile in PROFILES:
        write_profile(profile, base)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if fixtures/ is out of date")
    parser.add_argument("--out", type=Path, default=ROOT / "fixtures")
    args = parser.parse_args()
    if not args.check:
        generate(args.out)
        print(f"wrote {len(PROFILES)} fixture sets to {args.out}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        generate(Path(tmp))
        stale = [
            str(f.relative_to(tmp))
            for f in Path(tmp).rglob("*")
            if f.is_file()
            and (
                not (args.out / f.relative_to(tmp)).exists()
                or (args.out / f.relative_to(tmp)).read_text() != f.read_text()
            )
        ]
    if stale:
        print("stale fixtures (run scripts/make_fixtures.py):\n  " + "\n  ".join(stale))
        return 1
    print("fixtures are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
