"""Parsers for nvidia-smi, nccl-tests, nvbandwidth and DCGM output.

Each parser is deliberately tolerant: tool output drifts between versions, and a validation
tool that crashes on an unexpected column is worse than one that reports "not parsed".
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field

ANSI = re.compile(r"\x1b\[[0-9;]*m")

# --------------------------------------------------------------------------- nvidia-smi


@dataclass
class Gpu:
    index: int
    name: str
    bus_id: str
    memory_mib: int | None
    driver: str


def parse_gpu_query(text: str) -> list[Gpu]:
    """Parse `nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version`."""
    gpus: list[Gpu] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        mem = re.match(r"(\d+)\s*MiB", parts[3])
        gpus.append(
            Gpu(
                index=int(parts[0]),
                name=parts[1],
                bus_id=parts[2],
                memory_mib=int(mem.group(1)) if mem else None,
                driver=parts[4],
            )
        )
    return gpus


@dataclass
class TopoMatrix:
    devices: list[str]
    links: dict[tuple[str, str], str]
    cpu_affinity: dict[str, str] = field(default_factory=dict)
    numa_affinity: dict[str, str] = field(default_factory=dict)
    nic_names: dict[str, str] = field(default_factory=dict)

    @property
    def gpus(self) -> list[str]:
        return [d for d in self.devices if d.startswith("GPU")]

    @property
    def nics(self) -> list[str]:
        return [d for d in self.devices if not d.startswith("GPU")]

    def link(self, a: str, b: str) -> str:
        return self.links.get((a, b), "?")


DEVICE_RE = re.compile(r"^(GPU\d+|NIC\d+|mlx\d+_\d+)$")
# Lower is closer. NV# is handled separately.
PCIE_RANK = {"X": 0, "PIX": 2, "PXB": 3, "PHB": 4, "NODE": 5, "SYS": 6}


def link_rank(link: str) -> int:
    if link.startswith("NV"):
        return 1
    return PCIE_RANK.get(link, 9)


def nvlink_count(link: str) -> int | None:
    m = re.fullmatch(r"NV(\d+)", link)
    return int(m.group(1)) if m else None


def _cells(line: str) -> list[str]:
    if "\t" in line:
        return [c.strip() for c in line.split("\t") if c.strip()]
    return [c for c in re.split(r"\s{2,}", line.strip()) if c]


def parse_topo(text: str) -> TopoMatrix | None:
    """Parse `nvidia-smi topo -m`. Returns None if no matrix is found."""
    lines = ANSI.sub("", text).splitlines()
    header_idx = next((i for i, line in enumerate(lines) if re.search(r"\bGPU0\b", line)), None)
    if header_idx is None:
        return None
    header = _cells(lines[header_idx])
    devices = [c for c in header if DEVICE_RE.match(c)]
    extra_cols = [c for c in header if not DEVICE_RE.match(c)]
    topo = TopoMatrix(devices=devices, links={})

    for line in lines[header_idx + 1 :]:
        if not line.strip() or line.strip().startswith("Legend"):
            break
        cells = _cells(line)
        if not cells or cells[0] not in devices:
            continue
        name = cells[0]
        values = cells[1 : 1 + len(devices)]
        for other, value in zip(devices, values, strict=False):
            topo.links[(name, other)] = value
        extras = dict(zip(extra_cols, cells[1 + len(devices) :], strict=False))
        if "CPU Affinity" in extras:
            topo.cpu_affinity[name] = extras["CPU Affinity"]
        if "NUMA Affinity" in extras:
            topo.numa_affinity[name] = extras["NUMA Affinity"]

    in_nic_legend = False
    for line in lines:
        if line.strip().startswith("NIC Legend"):
            in_nic_legend = True
            continue
        if in_nic_legend:
            m = re.match(r"\s*(NIC\d+):\s*(\S+)", line)
            if m:
                topo.nic_names[m.group(1)] = m.group(2)
    return topo


@dataclass
class GpuFabric:
    bus_id: str
    product: str | None = None
    architecture: str | None = None
    state: str | None = None
    status: str | None = None
    clique_id: str | None = None
    cluster_uuid: str | None = None

    @property
    def in_multi_node_domain(self) -> bool:
        """True when this GPU has joined a rack-scale NVLink domain (NVL72-class)."""
        uuid = (self.cluster_uuid or "").lower().replace("-", "").replace("0", "")
        return (self.state or "").lower() == "completed" and uuid not in ("", "n/a", "na")


@dataclass
class SmiQuery:
    driver: str | None
    cuda: str | None
    gpus: list[GpuFabric]


def parse_smi_q(text: str) -> SmiQuery:
    """Pull driver/CUDA versions and each GPU's NVLink fabric state out of `nvidia-smi -q`."""
    driver = cuda = None
    gpus: list[GpuFabric] = []
    current: GpuFabric | None = None
    fabric_indent: int | None = None

    for raw in text.splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        gpu_header = re.match(r"^GPU\s+([0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d)$", line)
        if gpu_header and indent == 0:
            current = GpuFabric(bus_id=gpu_header.group(1))
            gpus.append(current)
            fabric_indent = None
            continue
        if fabric_indent is not None and indent <= fabric_indent:
            fabric_indent = None
        if current is not None and line == "Fabric":
            fabric_indent = indent
            continue
        if ":" not in line:
            continue
        key, _, value = (part.strip() for part in line.partition(":"))
        if current is None:
            if key == "Driver Version":
                driver = value
            elif key == "CUDA Version":
                cuda = value
            continue
        if fabric_indent is not None:
            attr = {
                "State": "state",
                "Status": "status",
                "CliqueId": "clique_id",
                "ClusterUUID": "cluster_uuid",
            }.get(key)
            if attr:
                setattr(current, attr, value)
        elif key == "Product Name":
            current.product = value
        elif key == "Product Architecture":
            current.architecture = value
    return SmiQuery(driver=driver, cuda=cuda, gpus=gpus)


def parse_dgx_release(text: str) -> dict[str, str]:
    """Parse KEY="value" lines from /etc/dgx-release."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r'\s*([A-Z0-9_]+)\s*=\s*"?(.*?)"?\s*$', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


# --------------------------------------------------------------------------- nccl-tests


def _num(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


@dataclass
class NcclRow:
    size: int
    count: int
    dtype: str
    redop: str
    root: int
    oop_time_us: float | None
    oop_algbw: float | None
    oop_busbw: float | None
    oop_wrong: int | None
    ip_time_us: float | None = None
    ip_algbw: float | None = None
    ip_busbw: float | None = None
    ip_wrong: int | None = None

    @property
    def best_busbw(self) -> float:
        return max(b for b in (self.oop_busbw, self.ip_busbw, 0.0) if b is not None)


@dataclass
class NcclResult:
    rows: list[NcclRow]
    avg_busbw: float | None = None
    nccl_version: str | None = None
    nranks: int | None = None
    out_of_bounds_ok: bool | None = None

    @property
    def peak(self) -> NcclRow | None:
        return max(self.rows, key=lambda r: r.best_busbw) if self.rows else None

    @property
    def wrong_total(self) -> int:
        return sum((r.oop_wrong or 0) + (r.ip_wrong or 0) for r in self.rows)


def _int(token: str) -> int | None:
    return int(token) if token.lstrip("-").isdigit() else None


def parse_nccl(text: str) -> NcclResult:
    """Parse one nccl-tests run (all_reduce_perf, alltoall_perf, ...)."""
    result = NcclResult(rows=[])
    ranks = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        version = re.search(r"NCCL version (\d+\.\d+\.\d+)", line)
        if version:
            result.nccl_version = version.group(1)
        if line.startswith("#"):
            if re.match(r"#\s+Rank\s+\d+", line):
                ranks += 1
            avg = re.match(r"#\s*Avg bus bandwidth\s*:\s*([\d.]+)", line)
            if avg:
                result.avg_busbw = float(avg.group(1))
            oob = re.match(r"#\s*Out of bounds values\s*:\s*(\d+)\s*(\w+)?", line)
            if oob:
                result.out_of_bounds_ok = oob.group(1) == "0"
            continue
        tokens = line.split()
        if len(tokens) < 9 or not tokens[0].isdigit():
            continue
        oop = tokens[5:9]
        ip = tokens[9:13] + ["N/A"] * (4 - len(tokens[9:13]))
        root = _int(tokens[4])
        result.rows.append(
            NcclRow(
                size=int(tokens[0]),
                count=int(tokens[1]) if tokens[1].isdigit() else 0,
                dtype=tokens[2],
                redop=tokens[3],
                root=root if root is not None else -1,
                oop_time_us=_num(oop[0]),
                oop_algbw=_num(oop[1]),
                oop_busbw=_num(oop[2]),
                oop_wrong=_int(oop[3]),
                ip_time_us=_num(ip[0]),
                ip_algbw=_num(ip[1]),
                ip_busbw=_num(ip[2]),
                ip_wrong=_int(ip[3]),
            )
        )
    result.nranks = ranks or None
    return result


# --------------------------------------------------------------------------- nvbandwidth


@dataclass
class NvbwTest:
    name: str
    sum_gbps: float | None = None
    # (row, column, GB/s). For device_to_device rows/columns are GPU indices; for
    # host_to_device the row is the CPU/NUMA node and the column the GPU.
    cells: list[tuple[int, int, float]] = field(default_factory=list)

    @property
    def avg_gbps(self) -> float | None:
        return statistics.fmean(v for _, _, v in self.cells) if self.cells else None

    @property
    def weakest(self) -> tuple[int, int, float] | None:
        return min(self.cells, key=lambda c: c[2]) if self.cells else None

    @property
    def median_gbps(self) -> float | None:
        return statistics.median(v for _, _, v in self.cells) if self.cells else None


@dataclass
class NvbwResult:
    tests: dict[str, NvbwTest]
    version: str | None = None


def parse_nvbandwidth(text: str) -> NvbwResult:
    result = NvbwResult(tests={})
    current: NvbwTest | None = None
    columns: list[int] = []
    for raw in text.splitlines():
        line = raw.strip()
        version = re.match(r"nvbandwidth Version:\s*(\S+)", line)
        if version:
            result.version = version.group(1)
            continue
        running = re.match(r"Running (\S+?)\.?$", line)
        if running:
            current = result.tests.setdefault(running.group(1), NvbwTest(running.group(1)))
            columns = []
            continue
        total = re.match(r"SUM (\S+) ([\d.]+)", line)
        if total:
            result.tests.setdefault(total.group(1), NvbwTest(total.group(1))).sum_gbps = float(
                total.group(2)
            )
            current = None
            continue
        if current is None:
            continue
        tokens = line.split()
        if tokens and all(t.isdigit() for t in tokens):
            columns = [int(t) for t in tokens]  # column header row
            continue
        if len(tokens) > 1 and tokens[0].isdigit() and columns:
            row = int(tokens[0])
            for col, token in zip(columns, tokens[1:], strict=False):
                value = _num(token)
                if value is not None:
                    current.cells.append((row, col, value))
    return result


# --------------------------------------------------------------------------- DCGM

STATUS_ORDER = {"skip": 0, "pass": 1, "warn": 2, "fail": 3}


def _norm_status(value: str) -> str:
    v = value.strip().lower()
    if v.startswith("pass"):
        return "pass"
    if v.startswith("warn"):
        return "warn"
    if v.startswith(("fail", "error")):
        return "fail"
    return "skip"


def worst(statuses) -> str:
    statuses = list(statuses)
    return max(statuses, key=STATUS_ORDER.__getitem__) if statuses else "skip"


@dataclass
class DcgmResult:
    tests: dict[str, str]
    messages: list[str]
    version: str | None = None
    error: str | None = None

    @property
    def overall(self) -> str:
        return worst(self.tests.values())


def _collect_messages(obj, out: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in ("warning", "warnings", "info", "message", "error"):
                if isinstance(value, str) and value.strip():
                    out.append(value.strip())
                else:
                    _collect_messages(value, out)
            elif isinstance(value, (dict, list)):
                _collect_messages(value, out)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            else:
                _collect_messages(item, out)


def parse_dcgm(text: str) -> DcgmResult:
    """Parse `dcgmi diag -j`. Walks the JSON tree for {"name": ..., "results": [...]} nodes,
    so it tolerates the schema changes between DCGM 3.x and 4.x."""
    start = text.find("{")
    if start < 0:
        return DcgmResult({}, [], error="no JSON object in dcgmi output")
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        return DcgmResult({}, [], error=f"invalid JSON: {exc}")

    tests: dict[str, str] = {}
    messages: list[str] = []

    def walk(obj) -> None:
        if isinstance(obj, dict):
            name, results = obj.get("name"), obj.get("results")
            if isinstance(name, str) and isinstance(results, list):
                statuses = [
                    _norm_status(r["status"])
                    for r in results
                    if isinstance(r, dict) and isinstance(r.get("status"), str)
                ]
                if statuses:
                    tests[name] = worst([tests.get(name, "skip"), *statuses])
                for r in results:
                    if isinstance(r, dict) and _norm_status(str(r.get("status", ""))) in (
                        "warn",
                        "fail",
                    ):
                        found: list[str] = []
                        _collect_messages(r, found)
                        gpu = r.get("gpu_id", r.get("entity_id"))
                        prefix = f"{name} (GPU {gpu})" if gpu is not None else name
                        messages.extend(f"{prefix}: {m}" for m in found)
            for key, value in obj.items():
                if key != "results":
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    version = data.get("version") if isinstance(data, dict) else None
    return DcgmResult(tests=tests, messages=messages, version=version)
