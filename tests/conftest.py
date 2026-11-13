from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"


@pytest.fixture
def fixture_dir():
    def get(name: str) -> Path:
        return FIXTURES / name

    return get


@pytest.fixture
def read():
    def get(profile: str, key: str) -> str:
        return (FIXTURES / profile / f"{key}.txt").read_text()

    return get
