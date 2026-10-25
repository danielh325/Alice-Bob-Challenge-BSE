import dataclasses
import math

import pytest

from catqubit.params import CatParams, to_angular

TWO_PI = 2 * math.pi


def test_to_angular_multiplies_by_two_pi():
    assert to_angular(1.0) == pytest.approx(TWO_PI)


def test_reglade_defaults_are_angular_frequencies_in_rad_per_us():
    p = CatParams.reglade()
    assert p.g2 == pytest.approx(TWO_PI * 0.763)
    assert p.kappa_b == pytest.approx(TWO_PI * 2.6)
    assert p.kappa_a == pytest.approx(TWO_PI * 9.3e-3)
    assert p.alpha == pytest.approx(2.1)


def test_buffer_drive_sets_alpha_squared():
    p = CatParams.reglade()
    assert p.eps_d == pytest.approx(p.alpha**2 * p.g2)
    assert p.eps_d / p.g2 == pytest.approx(4.41)


def test_kappa2_is_four_g2_squared_over_kappa_b():
    p = CatParams.reglade()
    assert p.kappa2 == pytest.approx(4 * p.g2**2 / p.kappa_b)
    assert p.kappa2 / TWO_PI == pytest.approx(4 * 0.763**2 / 2.6)


def test_adiabatic_ratios_show_eps_d_not_small_compared_with_kappa_b():
    g2_ratio, eps_ratio = CatParams.reglade().adiabatic_ratios()
    assert g2_ratio == pytest.approx(0.763 / 2.6)
    assert eps_ratio == pytest.approx(4.41 * 0.763 / 2.6)
    assert eps_ratio > 1


def test_params_are_immutable():
    p = CatParams.reglade()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.alpha = 1.0


def test_with_alpha_returns_new_instance_and_leaves_original_unchanged():
    p = CatParams.reglade()
    q = p.with_alpha(1.2)
    assert q.alpha == pytest.approx(1.2)
    assert q.eps_d == pytest.approx(1.44 * p.g2)
    assert p.alpha == pytest.approx(2.1)


@pytest.mark.parametrize("field", ["g2", "kappa_b"])
def test_non_positive_coupling_or_buffer_loss_rejected(field):
    kwargs = {"g2": 1.0, "kappa_b": 1.0, "kappa_a": 0.0, "alpha": 1.0, field: 0.0}
    with pytest.raises(ValueError, match=field):
        CatParams(**kwargs)


def test_negative_memory_loss_rejected():
    with pytest.raises(ValueError, match="kappa_a"):
        CatParams(g2=1.0, kappa_b=1.0, kappa_a=-1e-3, alpha=1.0)


def test_zero_memory_loss_allowed():
    assert CatParams(g2=1.0, kappa_b=1.0, kappa_a=0.0, alpha=1.0).kappa_a == 0.0


def test_non_positive_alpha_rejected():
    with pytest.raises(ValueError, match="alpha"):
        CatParams(g2=1.0, kappa_b=1.0, kappa_a=0.0, alpha=0.0)
