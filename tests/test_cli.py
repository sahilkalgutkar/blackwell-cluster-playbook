import json
import subprocess
import sys

from bcp.cli import main
from tests.conftest import ROOT


def test_topology_formats(capsys, fixture_dir):
    assert main(["topology", "--replay", str(fixture_dir("gb200-nvl72"))]) == 0
    assert "NVL72-class" in capsys.readouterr().out
    assert main(["topology", "--replay", str(fixture_dir("dgx-b200")), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["platform"]["dgx"]


def test_validate_exit_codes_and_report(tmp_path, capsys, fixture_dir):
    thresholds = str(ROOT / "configs" / "thresholds.yaml")
    args = ["validate", "--thresholds", thresholds, "--out", str(tmp_path)]
    assert main([*args, "--replay", str(fixture_dir("dgx-b200"))]) == 0
    assert main([*args, "--replay", str(fixture_dir("dgx-b200-degraded"))]) == 1
    runs = sorted(tmp_path.glob("*.json"))
    assert len(runs) == 2
    capsys.readouterr()
    assert main(["report", *map(str, runs)]) == 0
    table = capsys.readouterr().out
    assert "**FAIL**" in table and "**PASS**" in table and "*(synthetic)*" in table


def test_missing_replay_dir_is_a_clean_error(capsys):
    assert main(["topology", "--replay", "/nonexistent"]) == 2
    assert "not found" in capsys.readouterr().err


def test_fixtures_match_generator():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_fixtures.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
