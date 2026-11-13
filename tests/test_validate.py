from pathlib import Path

import pytest

from bcp.runner import ReplayRunner
from bcp.validate import load_thresholds, render_checks, run_validation
from tests.conftest import ROOT

THRESHOLDS = load_thresholds(ROOT / "configs" / "thresholds.yaml")


def validate(path: Path) -> dict:
    return run_validation(ReplayRunner(path), THRESHOLDS, progress=lambda _: None)


def by_name(doc: dict) -> dict:
    return {c["name"]: c for c in doc["checks"]}


@pytest.mark.parametrize("profile", ["dgx-b200", "hgx-h100", "gb200-nvl72"])
def test_healthy_fixtures_pass(fixture_dir, profile):
    doc = validate(fixture_dir(profile))
    assert doc["verdict"] == "pass", render_checks(doc)
    assert doc["run"]["synthetic"] is True
    assert set(doc["metrics"]["nccl"]) == {"all_reduce", "all_gather", "reduce_scatter", "alltoall"}


def test_degraded_fixture_fails_for_the_right_reasons(fixture_dir):
    doc = validate(fixture_dir("dgx-b200-degraded"))
    checks = by_name(doc)
    assert doc["verdict"] == "fail"
    assert checks["topology.nvlink_symmetry"]["status"] == "fail"
    assert checks["dcgm.diag"]["status"] == "fail"
    assert checks["nccl.all_reduce.busbw"]["status"] == "fail"
    assert checks["nccl.alltoall.busbw"]["status"] == "fail"
    assert checks["nccl.all_gather.busbw"]["status"] == "pass"


def test_nvl72_checks_fabric(fixture_dir):
    checks = by_name(validate(fixture_dir("gb200-nvl72")))
    assert checks["topology.nvlink_fabric"]["status"] == "pass"
    assert checks["inventory.gpu_count"]["measured"] == "4"


def test_missing_probes_are_skipped_not_fatal(tmp_path, fixture_dir):
    src = fixture_dir("dgx-b200")
    for key in ("gpu_query", "topo", "smi_q"):
        (tmp_path / f"{key}.txt").write_text((src / f"{key}.txt").read_text())
    doc = validate(tmp_path)
    checks = by_name(doc)
    assert checks["dcgm.diag"]["status"] == "skip"
    assert checks["nccl.all_reduce"]["status"] == "skip"
    assert doc["verdict"] == "pass"


def test_unknown_model_reports_info_not_failure(tmp_path, fixture_dir):
    src = fixture_dir("dgx-b200")
    for f in src.iterdir():
        (tmp_path / f.name).write_text(f.read_text().replace("NVIDIA B200", "NVIDIA X999"))
    doc = validate(tmp_path)
    assert by_name(doc)["nccl.all_reduce.busbw"]["status"] == "info"
    assert doc["verdict"] == "pass"


def test_render_checks_labels_synthetic(fixture_dir):
    text = render_checks(validate(fixture_dir("dgx-b200")))
    assert "SYNTHETIC" in text
    assert "nccl.all_reduce.busbw" in text
