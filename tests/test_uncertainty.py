import numpy as np
import pytest

from eoe_uq.uncertainty import decompose_uncertainty


def reference(e):
    n, m, _ = e.shape
    au = np.mean([np.mean(np.var(e[i], axis=0, ddof=1)) for i in range(n)])
    eu = np.mean(np.var(e.mean(axis=1), axis=0, ddof=1)) - au / m
    return au, eu


def test_matches_reference_formula():
    rng = np.random.default_rng(1)
    e = rng.normal(size=(7, 3, 10))
    out = decompose_uncertainty(e)
    au, eu = reference(e)
    assert out["aleatoric_uncertainty"] == pytest.approx(au)
    assert out["epistemic_uncertainty"] == pytest.approx(eu)
    assert out["total_uncertainty"] == pytest.approx(au + eu)


def test_identical_answers_have_zero_uncertainty():
    out = decompose_uncertainty(np.ones((5, 4, 8)))
    assert out["aleatoric_uncertainty"] == pytest.approx(0.0)
    assert out["epistemic_uncertainty"] == pytest.approx(0.0)


def test_no_within_perturbation_spread():
    rng = np.random.default_rng(0)
    e = np.repeat(rng.normal(size=(6, 1, 16)), 5, axis=1)
    out = decompose_uncertainty(e)
    assert out["aleatoric_uncertainty"] == pytest.approx(0.0)
    assert out["epistemic_uncertainty"] > 0


def test_same_inner_set_for_every_perturbation():
    rng = np.random.default_rng(0)
    e = np.repeat(rng.normal(size=(1, 5, 16)), 6, axis=0)
    out = decompose_uncertainty(e)
    assert out["aleatoric_uncertainty"] > 0
    assert out["epistemic_uncertainty"] == pytest.approx(-out["aleatoric_uncertainty"] / 5)


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        decompose_uncertainty(np.zeros((4, 8)))
    with pytest.raises(ValueError):
        decompose_uncertainty(np.zeros((1, 4, 8)))
