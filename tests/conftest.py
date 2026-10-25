import dynamiqs as dq
import pytest

from catqubit.params import CatParams


def pytest_configure(config: pytest.Config) -> None:
    # Section 0.5 of the challenge: 32-bit floats are too coarse for these decays.
    dq.set_precision("double")


@pytest.fixture
def params() -> CatParams:
    """Provided parameters (Réglade et al., Table S2) in rad/us."""
    return CatParams.reglade()
