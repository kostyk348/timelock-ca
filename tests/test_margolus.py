import numpy as np
import pytest

from timelock_ca.margolus import (BBM_FWD, BBM_REV, meet_in_the_middle,
                                  run_backward, run_forward, step_margolus,
                                  verify_reversibility)


def test_bbm_is_bijection():
    assert sorted(BBM_FWD.tolist()) == list(range(16))
    for i in range(16):
        assert BBM_REV[BBM_FWD[i]] == i


def test_hamming_weight_invariant():
    rng = np.random.default_rng(0)
    grid = (rng.random((16, 16)) < 0.3).astype(np.uint8)
    w0 = int(grid.sum())
    g = grid
    for t in range(60):
        g = step_margolus(g, phase=t % 2)
    assert int(g.sum()) == w0


def test_bit_exact_reversibility():
    report = verify_reversibility(size=16, steps=100, seed=0)
    assert report["entropy_conserved"]
    assert report["bit_exact_reversal"]


def test_odd_dimensions_rejected():
    with pytest.raises(ValueError):
        step_margolus(np.zeros((7, 8), dtype=np.uint8), phase=0)


def test_meet_in_the_middle():
    mid = meet_in_the_middle(size=32, steps=40, seed=1337)
    assert mid["endpoint_recovered"]
    assert mid["midpoint_match"]


def test_forward_backward_helpers_match():
    rng = np.random.default_rng(5)
    grid = (rng.random((12, 12)) < 0.4).astype(np.uint8)
    assert np.array_equal(run_backward(run_forward(grid, 37), 37), grid)
