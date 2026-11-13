"""Runners execute probes live or replay previously collected output."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from bcp.probes import Probe

MANIFEST = "manifest.json"


@dataclass
class ProbeOutput:
    key: str
    text: str | None
    exit_code: int | None = None
    duration_s: float | None = None
    note: str = ""

    @property
    def available(self) -> bool:
        return self.text is not None and self.text.strip() != ""


class Runner(Protocol):
    def run(self, probe: Probe, params: dict[str, str]) -> ProbeOutput: ...

    def metadata(self) -> dict: ...


@dataclass
class LiveRunner:
    """Runs probes on this machine and, if `save_dir` is set, archives raw output."""

    save_dir: Path | None = None
    _manifest: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._manifest = {
            "schema": 1,
            "host": socket.gethostname(),
            "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "synthetic": False,
            "probes": {},
        }
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def run(self, probe: Probe, params: dict[str, str]) -> ProbeOutput:
        argv = probe.render(params)
        if probe.optional and argv[0] == "cat" and not Path(argv[1]).exists():
            return self._record(probe, argv, ProbeOutput(probe.key, None, note="not present"))
        if shutil.which(argv[0]) is None:
            return self._record(
                probe, argv, ProbeOutput(probe.key, None, note=f"{argv[0]} not found on PATH")
            )
        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=probe.timeout_s,
                env={**os.environ, **dict(probe.env)},
                check=False,
            )
        except subprocess.TimeoutExpired:
            out = ProbeOutput(probe.key, None, note=f"timed out after {probe.timeout_s}s")
            return self._record(probe, argv, out)
        # Keep stdout even on a non-zero exit: dcgmi exits non-zero when a test fails,
        # and that JSON is exactly what we want to see.
        out = ProbeOutput(
            probe.key,
            proc.stdout,
            exit_code=proc.returncode,
            duration_s=round(time.monotonic() - start, 2),
            note=proc.stderr.strip()[-500:] if proc.returncode else "",
        )
        return self._record(probe, argv, out)

    def _record(self, probe: Probe, argv: list[str], out: ProbeOutput) -> ProbeOutput:
        self._manifest["probes"][probe.key] = {
            "argv": argv,
            "exit_code": out.exit_code,
            "duration_s": out.duration_s,
            "note": out.note,
        }
        if self.save_dir:
            if out.text is not None:
                (self.save_dir / f"{probe.key}.txt").write_text(out.text)
            (self.save_dir / MANIFEST).write_text(json.dumps(self._manifest, indent=2) + "\n")
        return out

    def metadata(self) -> dict:
        meta = {k: v for k, v in self._manifest.items() if k != "probes"}
        meta["source"] = "live"
        return meta


@dataclass
class ReplayRunner:
    """Reads `<key>.txt` files written by `bcp collect` (or the fixture generator)."""

    directory: Path

    def __post_init__(self) -> None:
        if not self.directory.is_dir():
            raise FileNotFoundError(f"replay directory not found: {self.directory}")
        manifest_path = self.directory / MANIFEST
        self._manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def run(self, probe: Probe, params: dict[str, str]) -> ProbeOutput:
        path = self.directory / f"{probe.key}.txt"
        if not path.exists():
            return ProbeOutput(probe.key, None, note="not in replay directory")
        recorded = self._manifest.get("probes", {}).get(probe.key, {})
        return ProbeOutput(
            probe.key,
            path.read_text(),
            exit_code=recorded.get("exit_code"),
            duration_s=recorded.get("duration_s"),
        )

    def metadata(self) -> dict:
        meta = {k: v for k, v in self._manifest.items() if k != "probes"}
        meta.setdefault("host", self.directory.name)
        meta.setdefault("collected_at", None)
        meta.setdefault("synthetic", False)
        meta["source"] = f"replay:{self.directory}"
        return meta
