"""Every command bcp runs on a node, in one place.

`bcp collect` runs these and saves each stdout as `<key>.txt`. `bcp validate --replay DIR`
reads the same files back, so a session on rented hardware can be analysed offline later,
and the synthetic fixtures under `fixtures/` exercise exactly the same code path in CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

SUITES: dict[str, dict[str, str]] = {
    # ~5 minutes on an 8-GPU node. Use it as a smoke test before a long job.
    "quick": {"dcgm_level": "1", "nccl_max": "1G"},
    # ~30-60 minutes. Use it for burn-in / acceptance of a new node.
    "full": {"dcgm_level": "3", "nccl_max": "16G"},
}

NCCL_OPS: dict[str, str] = {
    "nccl_all_reduce": "all_reduce",
    "nccl_all_gather": "all_gather",
    "nccl_reduce_scatter": "reduce_scatter",
    "nccl_alltoall": "alltoall",
}

NVBANDWIDTH_TESTS: tuple[str, ...] = (
    "device_to_device_memcpy_read_ce",
    "host_to_device_memcpy_ce",
    "device_to_host_memcpy_ce",
)


@dataclass(frozen=True)
class Probe:
    key: str
    argv: tuple[str, ...]
    description: str
    timeout_s: int = 600
    env: tuple[tuple[str, str], ...] = ()
    # Absent on most machines by design (e.g. /etc/dgx-release outside DGX OS).
    optional: bool = False

    def render(self, params: dict[str, str]) -> list[str]:
        return [part.format(**params) for part in self.argv]


def _nccl(key: str, binary: str) -> Probe:
    return Probe(
        key=key,
        argv=(
            "{nccl_bin}" + binary,
            "-b",
            "8",
            "-e",
            "{nccl_max}",
            "-f",
            "2",
            "-g",
            "{ngpus}",
        ),
        description=f"nccl-tests {binary} across all local GPUs",
        timeout_s=1800,
        env=(("NCCL_DEBUG", "VERSION"),),
    )


PROBES: tuple[Probe, ...] = (
    Probe(
        "gpu_query",
        (
            "nvidia-smi",
            "--query-gpu=index,name,pci.bus_id,memory.total,driver_version",
            "--format=csv,noheader",
        ),
        "GPU inventory",
        timeout_s=60,
    ),
    Probe("topo", ("nvidia-smi", "topo", "-m"), "GPU/NIC topology matrix", timeout_s=60),
    Probe("smi_q", ("nvidia-smi", "-q"), "Device query incl. NVLink fabric state", timeout_s=120),
    Probe(
        "dgx_release",
        ("cat", "/etc/dgx-release"),
        "DGX OS release file",
        timeout_s=10,
        optional=True,
    ),
    Probe(
        "dcgm_diag",
        ("dcgmi", "diag", "-r", "{dcgm_level}", "-j"),
        "DCGM diagnostics",
        timeout_s=3600,
    ),
    _nccl("nccl_all_reduce", "all_reduce_perf"),
    _nccl("nccl_all_gather", "all_gather_perf"),
    _nccl("nccl_reduce_scatter", "reduce_scatter_perf"),
    _nccl("nccl_alltoall", "alltoall_perf"),
    Probe(
        "nvbandwidth",
        ("nvbandwidth", *[arg for t in NVBANDWIDTH_TESTS for arg in ("-t", t)]),
        "nvbandwidth copy-engine bandwidth",
        timeout_s=1800,
    ),
)

PROBE_BY_KEY: dict[str, Probe] = {p.key: p for p in PROBES}


def base_params(suite: str) -> dict[str, str]:
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choose from {', '.join(SUITES)}")
    nccl_dir = os.environ.get("BCP_NCCL_TESTS_DIR", "").rstrip("/")
    return {**SUITES[suite], "nccl_bin": f"{nccl_dir}/" if nccl_dir else "", "ngpus": "1"}
