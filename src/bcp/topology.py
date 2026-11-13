"""Work out what kind of node this is (HGX baseboard, NVL72 tray, DGX, ...) and how it is wired."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field

from bcp.parsers import (
    Gpu,
    SmiQuery,
    TopoMatrix,
    link_rank,
    nvlink_count,
    parse_dgx_release,
    parse_gpu_query,
    parse_smi_q,
    parse_topo,
)

# Order matters: "GB200" must match before "B200", "GH200" before "H200".
# (substring in the GPU name, model key, generation, CPU+GPU superchip)
GPU_MODELS: tuple[tuple[str, str, str, bool], ...] = (
    ("VR200", "VR200", "Rubin", True),  # MLPerf designation; confirm the nvidia-smi string
    ("GB300", "GB300", "Blackwell Ultra", True),
    ("B300", "B300", "Blackwell Ultra", False),
    ("GB200", "GB200", "Blackwell", True),
    ("B200", "B200", "Blackwell", False),
    ("GB10", "GB10", "Blackwell", True),
    ("GH200", "GH200", "Hopper", True),
    ("H200", "H200", "Hopper", False),
    ("H100", "H100", "Hopper", False),
)

# Per-GPU NVLink bandwidth, bidirectional, GB/s (NVLink 4 / 5 / 6).
NVLINK_GBPS = {"Hopper": 900, "Blackwell": 1800, "Blackwell Ultra": 1800, "Rubin": 3600}

# SXM parts on HGX / NVL72 expose 18 NVLinks per GPU.
EXPECTED_NVLINKS = {"H100": 18, "H200": 18, "B200": 18, "B300": 18, "GB200": 18, "GB300": 18}


def identify(name: str, architecture: str | None = None) -> tuple[str, str, bool]:
    upper = name.upper()
    for needle, key, generation, superchip in GPU_MODELS:
        if needle in upper:
            return key, generation, superchip
    return "unknown", architecture or "unknown", False


@dataclass
class Platform:
    gpu_model: str
    gpu_name: str
    generation: str
    gpu_count: int
    memory_mib: int | None
    superchip: bool
    form_factor: str
    nvlink_links: int | None
    nvlink_symmetric: bool | None
    multi_node_nvlink: bool
    clique_id: str | None
    cluster_uuid: str | None
    fabric_state: str | None
    dgx: str | None
    driver: str | None
    cuda: str | None
    nominal_nvlink_gbps: int | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Topology:
    platform: Platform
    matrix: TopoMatrix | None
    nearest_nic: dict[str, list[str]]
    weak_nvlink_gpus: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "platform": asdict(self.platform),
            "nearest_nic": self.nearest_nic,
            "weak_nvlink_gpus": self.weak_nvlink_gpus,
            "links": {
                f"{a}-{b}": v
                for (a, b), v in (self.matrix.links if self.matrix else {}).items()
                if a < b
            },
        }


def analyze(
    gpu_query: str | None,
    topo: str | None,
    smi_q: str | None,
    dgx_release: str | None = None,
) -> Topology:
    gpus: list[Gpu] = parse_gpu_query(gpu_query) if gpu_query else []
    matrix = parse_topo(topo) if topo else None
    query = parse_smi_q(smi_q) if smi_q else SmiQuery(None, None, [])
    dgx = parse_dgx_release(dgx_release) if dgx_release else {}

    name = gpus[0].name if gpus else (query.gpus[0].product if query.gpus else None) or "unknown"
    arch = query.gpus[0].architecture if query.gpus else None
    model, generation, superchip = identify(name, arch)
    count = len(gpus) or (len(matrix.gpus) if matrix else 0) or len(query.gpus)
    warnings: list[str] = []

    # NVLink between GPU pairs.
    pair_links: dict[tuple[str, str], int] = {}
    if matrix:
        for a in matrix.gpus:
            for b in matrix.gpus:
                if a < b:
                    n = nvlink_count(matrix.link(a, b))
                    if n is not None:
                        pair_links[(a, b)] = n
    gpu_pairs = len(matrix.gpus) * (len(matrix.gpus) - 1) // 2 if matrix else 0
    nv_all = bool(pair_links) and len(pair_links) == gpu_pairs
    mode_links = Counter(pair_links.values()).most_common(1)[0][0] if pair_links else None
    expected = EXPECTED_NVLINKS.get(model)
    target = max(filter(None, (mode_links, expected)), default=None)

    weak: dict[str, int] = {}
    if target:
        for (a, b), n in pair_links.items():
            if n < target:
                for gpu in (a, b):
                    weak[gpu] = min(weak.get(gpu, n), n)
        # A GPU that is low on every pair is the degraded one; drop the healthy partners.
        if weak and len(weak) > 1:
            per_gpu = Counter(g for (a, b), n in pair_links.items() if n < target for g in (a, b))
            top = max(per_gpu.values())
            weak = {g: n for g, n in weak.items() if per_gpu[g] == top}
        for gpu, n in sorted(weak.items()):
            warnings.append(f"{gpu} has NV{n} to some peers, expected NV{target}: degraded NVLink?")
    symmetric = None if not pair_links else (len(set(pair_links.values())) == 1 and nv_all)
    if pair_links and not nv_all:
        warnings.append("Not every GPU pair is NVLink-connected")

    # Rack-scale NVLink fabric (NVL72-class) shows up per GPU in `nvidia-smi -q`.
    fabric = query.gpus[0] if query.gpus else None
    multi_node = bool(query.gpus) and all(g.in_multi_node_domain for g in query.gpus)
    if superchip and model != "GB10" and fabric and (fabric.state or "").lower() != "completed":
        warnings.append(
            f"NVLink fabric state is {fabric.state!r}: the NVLink domain has not formed "
            "(check NVLink Switch trays / NMX and the IMEX daemon)"
        )

    if model == "GB10":
        form = "DGX Spark-class desktop (GB10)"
    elif multi_node:
        form = f"NVL72-class rack node ({count} GPUs here, multi-node NVLink domain)"
    elif superchip and nv_all:
        form = f"Grace superchip node ({count} GPUs, no multi-node NVLink domain)"
    elif count == 8 and nv_all:
        form = "HGX 8-GPU baseboard (NVSwitch)"
    elif pair_links:
        form = f"NVLink-connected {count}-GPU node"
    elif count:
        form = f"PCIe {count}-GPU node"
    else:
        form = "unknown"

    nearest: dict[str, list[str]] = {}
    if matrix and matrix.nics:
        for gpu in matrix.gpus:
            ranked = sorted(matrix.nics, key=lambda nic, g=gpu: link_rank(matrix.link(g, nic)))
            best = link_rank(matrix.link(gpu, ranked[0]))
            nearest[gpu] = [
                f"{nic} ({matrix.nic_names.get(nic, nic)}, {matrix.link(gpu, nic)})"
                for nic in ranked
                if link_rank(matrix.link(gpu, nic)) == best
            ]
            if best >= link_rank("SYS"):
                warnings.append(f"{gpu} has no NIC on its socket: RDMA traffic will cross CPUs")

    dgx_name = dgx.get("DGX_PLATFORM") or dgx.get("DGX_PRETTY_NAME") or dgx.get("DGX_NAME")
    platform = Platform(
        gpu_model=model,
        gpu_name=name,
        generation=generation,
        gpu_count=count,
        memory_mib=gpus[0].memory_mib if gpus else None,
        superchip=superchip,
        form_factor=form,
        nvlink_links=mode_links,
        nvlink_symmetric=symmetric,
        multi_node_nvlink=multi_node,
        clique_id=fabric.clique_id if fabric else None,
        cluster_uuid=fabric.cluster_uuid if fabric else None,
        fabric_state=fabric.state if fabric else None,
        dgx=dgx_name,
        driver=(gpus[0].driver if gpus else None) or query.driver,
        cuda=query.cuda,
        nominal_nvlink_gbps=NVLINK_GBPS.get(generation) if pair_links else None,
        warnings=warnings,
    )
    return Topology(platform=platform, matrix=matrix, nearest_nic=nearest, weak_nvlink_gpus=weak)


# --------------------------------------------------------------------------- rendering


def render_text(topology: Topology) -> str:
    p = topology.platform
    lines = [
        f"Platform      : {p.form_factor}" + (f" · {p.dgx}" if p.dgx else ""),
        f"GPUs          : {p.gpu_count} x {p.gpu_name} ({p.generation})"
        + (f", {p.memory_mib} MiB each" if p.memory_mib else ""),
        f"Driver / CUDA : {p.driver or '?'} / {p.cuda or '?'}",
    ]
    if p.nvlink_links:
        sym = "symmetric" if p.nvlink_symmetric else "ASYMMETRIC"
        lines.append(
            f"NVLink        : NV{p.nvlink_links} per GPU pair ({sym})"
            + (f" · nominal {p.nominal_nvlink_gbps} GB/s per GPU" if p.nominal_nvlink_gbps else "")
        )
    else:
        lines.append("NVLink        : none detected")
    if p.fabric_state:
        domain = (
            f"multi-node domain, clique {p.clique_id}, cluster {p.cluster_uuid}"
            if p.multi_node_nvlink
            else "single node"
        )
        lines.append(f"Fabric        : {p.fabric_state} ({domain})")
    if topology.nearest_nic:
        lines.append("Nearest NIC   :")
        lines += [f"  {gpu:<5} -> {', '.join(nics)}" for gpu, nics in topology.nearest_nic.items()]
    lines.append("Warnings      : " + ("none" if not p.warnings else ""))
    lines += [f"  ! {w}" for w in p.warnings]
    return "\n".join(lines)


def render_json(topology: Topology) -> str:
    return json.dumps(topology.to_dict(), indent=2)


def render_mermaid(topology: Topology) -> str:
    """A Mermaid flowchart; GitHub renders it inline in Markdown."""
    p, m = topology.platform, topology.matrix
    out = ["flowchart LR"]
    if not m:
        out.append('  none["no topology matrix"]')
        return "\n".join(out)

    by_numa: dict[str, list[str]] = {}
    for gpu in m.gpus:
        by_numa.setdefault(m.numa_affinity.get(gpu, "?"), []).append(gpu)
    # NICs have no NUMA column in `topo -m`; place each next to its closest GPU.
    for nic in m.nics:
        closest = min(m.gpus, key=lambda g, n=nic: link_rank(m.link(g, n)), default=None)
        numa = m.numa_affinity.get(closest, "?") if closest else "?"
        by_numa.setdefault(numa, []).append(nic)
    for numa, devs in by_numa.items():
        cpus = next((m.cpu_affinity[d] for d in devs if d in m.cpu_affinity), "")
        label = f"NUMA {numa}" + (f" · CPUs {cpus}" if cpus else "")
        out.append(f'  subgraph numa{numa.replace("?", "x")}["{label}"]')
        for dev in devs:
            if dev.startswith("GPU"):
                out.append(f'    {dev}["{dev}<br/>{p.gpu_model}"]')
            else:
                out.append(f'    {dev}(["{dev}<br/>{m.nic_names.get(dev, "")}"])')
        out.append("  end")

    if p.nvlink_links:
        fabric_label = (
            f"NVLink Switch trays<br/>rack-wide domain, clique {p.clique_id}"
            if p.multi_node_nvlink
            else "NVSwitch"
        )
        out.append(f'  FABRIC{{{{"{fabric_label}"}}}}')
        for gpu in m.gpus:
            links = topology.weak_nvlink_gpus.get(gpu, p.nvlink_links)
            out.append(f"  {gpu} ===|NV{links}| FABRIC")
    for nic in m.nics:
        for gpu in m.gpus:
            link = m.link(gpu, nic)
            if link in ("PIX", "PXB"):
                out.append(f"  {gpu} ---|{link}| {nic}")
    if topology.weak_nvlink_gpus:
        out.append("  classDef warn fill:#fde2e1,stroke:#c0392b,stroke-width:2px")
        out.append(f"  class {','.join(sorted(topology.weak_nvlink_gpus))} warn")
    return "\n".join(out)
