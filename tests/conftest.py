import pytest
from agents import set_tracing_disabled


@pytest.fixture(autouse=True)
def disable_hosted_tracing() -> None:
    set_tracing_disabled(True)
