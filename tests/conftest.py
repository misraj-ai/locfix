from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TRACES = FIXTURES / "traces"


def trace_path(prefix: str) -> Path:
    matches = sorted(TRACES.glob(prefix + "*.json"))
    assert len(matches) == 1, (prefix, matches)
    return matches[0]


@pytest.fixture
def exceptiongroup_path() -> Path:
    return trace_path("agronholm__exceptiongroup")


@pytest.fixture
def arrow_path() -> Path:
    return trace_path("arrow-py__arrow")


@pytest.fixture
def astropy_path() -> Path:
    return trace_path("astropy__astropy")


@pytest.fixture
def gunicorn_path() -> Path:
    return trace_path("benoitc__gunicorn")


@pytest.fixture
def all_trace_paths() -> list[Path]:
    return sorted(TRACES.glob("*.json"))
