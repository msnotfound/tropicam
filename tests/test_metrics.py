"""Metrics and harness behave as the experiments assume.

These guard the measurement layer itself -- if a metric is wrong, every
downstream number is wrong in a way no amount of engine correctness fixes.

Runs under pytest, or standalone: ``python3 tests/test_metrics.py``
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tropicam.harness import Record, ResultLog, Sweep, rng_for  # noqa: E402
from tropicam.metrics import (  # noqa: E402
    angular_error, endpoint_error, normal_flow_error,
    regime_agreement, check_certificate, contrast_rate_floor,
)
from tropicam.synthetic import translating_corner  # noqa: E402


def test_angular_error_zero_on_identical_flow():
    f = np.array([[[1.0, 2.0], [0.0, 0.0]]])
    assert np.allclose(angular_error(f, f), 0.0, atol=1e-9)


def test_angular_error_ninety_degrees():
    a = np.array([[1e6, 0.0]])
    b = np.array([[0.0, 1e6]])
    assert abs(angular_error(a, b)[0] - 90.0) < 1e-3


def test_endpoint_error_is_l2():
    a = np.array([[3.0, 4.0]])
    b = np.array([[0.0, 0.0]])
    assert np.allclose(endpoint_error(a, b), 5.0)


def test_normal_flow_error_ignores_tangential_component():
    """The bar scene's whole point: v_y is unconstrained, not wrong."""
    est = np.array([[0.05, 999.0]])   # wildly wrong tangential component
    gt = np.array([[0.05, 0.0]])
    assert np.allclose(normal_flow_error(est, gt, normal=(1.0, 0.0)), 0.0)
    # ... whereas endpoint error would call this a catastrophic failure.
    assert endpoint_error(est, gt)[0] > 900


def test_regime_agreement_rewards_refinement_not_count_match():
    """Two regimes splitting one object is CORRECT, not a 2x over-count.

    This is the H5 correction: section 6.2 says a crease is a velocity change,
    not a new object, so an object may legitimately contain several regimes.
    Purity must stay 1.0 here even though the counts disagree 2:1.
    """
    objects = np.zeros((10, 10), dtype=int)
    regimes = np.zeros((10, 10), dtype=int)
    regimes[5:] = 1                      # one object, two regimes

    r = regime_agreement(regimes, objects)
    assert r.purity == 1.0
    assert r.impure_regimes == 0
    assert r.n_regimes == 2 and r.n_objects == 1
    assert r.count_ratio == 2.0          # descriptive only


def test_regime_agreement_penalises_straddling():
    """A regime crossing an object boundary is the real failure."""
    objects = np.zeros((10, 10), dtype=int)
    objects[:, 5:] = 1
    regimes = np.zeros((10, 10), dtype=int)   # one regime over two objects

    r = regime_agreement(regimes, objects)
    assert r.impure_regimes == 1
    assert r.purity == 0.0


def test_regime_agreement_ignores_unlabelled():
    objects = np.full((6, 6), -1, dtype=int)
    regimes = np.full((6, 6), -1, dtype=int)
    objects[:3] = 0
    regimes[:3] = 0
    r = regime_agreement(regimes, objects)
    assert r.purity == 1.0 and r.n_objects == 1


def test_certificate_holds_and_reports_tightness():
    dev = np.array([0.1, 0.2, 0.35])
    c = check_certificate(dev, bound=0.4)
    assert c.holds and c.n_violations == 0
    assert abs(c.tightness - 0.875) < 1e-9


def test_certificate_detects_violation():
    c = check_certificate(np.array([0.1, 0.9]), bound=0.5)
    assert not c.holds and c.n_violations == 1
    assert c.slack < 0


def test_vacuous_bound_holds_but_is_flagged_by_tightness():
    """A bound that holds proves nothing on its own -- infinity holds too."""
    c = check_certificate(np.array([0.01, 0.02]), bound=1e6)
    assert c.holds
    assert c.tightness < 1e-6, "vacuous bound must be visible as low tightness"


def test_contrast_rate_floor_is_finite_on_a_real_surface():
    scene = translating_corner()
    valid = np.isfinite(scene.analytic_surface)
    r = contrast_rate_floor(np.nan_to_num(scene.analytic_surface), valid)
    assert np.isfinite(r) and r > 0


def test_rng_for_is_deterministic_and_condition_independent():
    """Adding a condition must not shift existing conditions' noise draws.

    Otherwise every previously recorded number silently changes when a sweep
    is extended.
    """
    a1 = rng_for(0, "e1", "rate=1.0").normal(size=5)
    a2 = rng_for(0, "e1", "rate=1.0").normal(size=5)
    b = rng_for(0, "e1", "rate=2.0").normal(size=5)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)


def test_result_log_roundtrip_and_claim_tracking():
    with tempfile.TemporaryDirectory() as d:
        log = ResultLog(Path(d) / "results.jsonl")
        log.append(Record(experiment="e1", claim="C1", params={"rate": 1.0},
                          metrics={"ae": 3.2}, seed=0))
        log.append(Record(experiment="e2", claim="C2", params={},
                          metrics={"purity": 0.9}, seed=0))
        rows = log.load()
        assert len(rows) == 2
        assert log.claims_covered() == {"C1", "C2"}
        assert log.filter(experiment="e1")[0]["metrics"]["ae"] == 3.2
        assert rows[0]["git"] and rows[0]["env"]["numpy"]


def test_sweep_records_every_condition():
    with tempfile.TemporaryDirectory() as d:
        log = ResultLog(Path(d) / "s.jsonl")
        sweep = Sweep(experiment="e1", claim="C1", log=log, seed=7)
        conds = [{"rate": r} for r in (0.1, 1.0, 10.0)]
        recs = sweep.run(conds, lambda p, rng: {"x": float(rng.normal())})
        assert len(recs) == 3
        assert len(log.load()) == 3
        assert {r["params"]["rate"] for r in log.load()} == {0.1, 1.0, 10.0}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
