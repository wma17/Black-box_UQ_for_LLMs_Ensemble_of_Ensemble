import numpy as np
import pytest

from eoe_uq.uncertainty import decompose_uncertainty


def test_identical_answers_have_zero_uncertainty():
    e = np.ones((5, 4, 8))
    out = decompose_uncertainty(e)
    assert out["aleatoric_uncertainty"] == pytest.approx(0.0)
    assert out["epistemic_uncertainty"] == pytest.approx(0.0)


def test_only_between_perturbation_spread_is_epistemic():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(6, 1, 16))   # differs per outer sample
    e = np.repeat(centers, 5, axis=1)       # identical inside each outer sample
    out = decompose_uncertainty(e)
    assert out["aleatoric_uncertainty"] == pytest.approx(0.0)
    assert out["epistemic_uncertainty"] > 0


def test_only_within_perturbation_spread_is_aleatoric():
    rng = np.random.default_rng(0)
    inner = rng.normal(size=(1, 5, 16))
    e = np.repeat(inner, 6, axis=0)         # same inner set for every outer sample
    out = decompose_uncertainty(e)
    assert out["epistemic_uncertainty"] == pytest.approx(0.0)
    assert out["aleatoric_uncertainty"] > 0


def test_matches_reference_formula():
    rng = np.random.default_rng(1)
    e = rng.normal(size=(7, 3, 10))
    out = decompose_uncertainty(e)
    au = np.mean([np.mean(np.var(e[n], axis=0, ddof=1)) for n in range(7)])
    eu = np.mean(np.var(e.mean(axis=1), axis=0, ddof=1))
    assert out["aleatoric_uncertainty"] == pytest.approx(au)
    assert out["epistemic_uncertainty"] == pytest.approx(eu)
    assert out["total_uncertainty"] == pytest.approx(au + eu)


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        decompose_uncertainty(np.zeros((4, 8)))
    with pytest.raises(ValueError):
        decompose_uncertainty(np.zeros((1, 4, 8)))
