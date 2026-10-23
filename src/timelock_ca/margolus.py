"""Margolus / BBM block cellular automaton — bit-exact time travel.

The Billiard Ball Model partitioned into 2x2 blocks. Every block transition
is a *permutation* of the 16 possible states: Hamming weight (particle
number) is strictly invariant, the rule is its own inverse family, and
running time backwards is just the inverse LUT. Two-phase (odd/even) block
offsets give the neighbourhood its dynamics.
"""

from __future__ import annotations

import numpy as np

# Billiard-ball permutation table: 16 block states -> 16 block states.
# 0001->0010->0100->1000->0001 (single particle rotates: momentum kept)
# 1001 <-> 0110                (elastic two-particle collision)
BBM_FWD = np.array([0, 8, 4, 3, 2, 10, 9, 14, 1, 6, 5, 13, 12, 11, 7, 15],
                   dtype=np.uint8)
BBM_REV = np.zeros(16, dtype=np.uint8)
for _i in range(16):
    BBM_REV[BBM_FWD[_i]] = _i


def step_margolus(grid: np.ndarray, phase: int, reverse: bool = False) -> np.ndarray:
    """One tick of the reversible block machine.

    grid   : 2D uint8 array (even dimensions)
    phase  : 0 or 1 — Margolus lattice offset
    reverse: use the inverse table (run time backwards)
    """
    lut = BBM_REV if reverse else BBM_FWD
    H, W = grid.shape
    if H % 2 or W % 2:
        raise ValueError("margolus grid needs even dimensions")

    if phase == 1:
        grid = np.roll(grid, (-1, -1), axis=(0, 1))

    tl = grid[0::2, 0::2]
    tr = grid[0::2, 1::2]
    bl = grid[1::2, 0::2]
    br = grid[1::2, 1::2]

    idx = (tl << 3) | (tr << 2) | (bl << 1) | br
    out = lut[idx]

    grid[0::2, 0::2] = (out >> 3) & 1
    grid[0::2, 1::2] = (out >> 2) & 1
    grid[1::2, 0::2] = (out >> 1) & 1
    grid[1::2, 1::2] = out & 1

    if phase == 1:
        grid = np.roll(grid, (1, 1), axis=(0, 1))
    return grid


def run_forward(grid: np.ndarray, steps: int) -> np.ndarray:
    g = grid.copy()
    for t in range(steps):
        g = step_margolus(g, phase=t % 2)
    return g


def run_backward(grid: np.ndarray, steps: int) -> np.ndarray:
    g = grid.copy()
    for t in reversed(range(steps)):
        g = step_margolus(g, phase=t % 2, reverse=True)
    return g


def verify_reversibility(size: int = 16, steps: int = 100,
                         seed: int = 0) -> dict[str, bool]:
    """Forward N ticks, then backward N ticks: must land bit-exactly on start."""
    rng = np.random.default_rng(seed)
    grid = (rng.random((size, size)) < 0.2).astype(np.uint8)
    initial_bits = int(grid.sum())

    g = run_forward(grid, steps)
    entropy_kept = int(g.sum()) == initial_bits

    g = run_backward(g, steps)
    return {
        "entropy_conserved": entropy_kept,
        "bit_exact_reversal": bool(np.array_equal(g, grid)),
    }


def meet_in_the_middle(size: int = 32, steps: int = 40, seed: int = 1337) -> dict:
    """Symmetric time convergence.

    Given S(0), compute S(T) forward and simultaneously reconstruct S(0)
    backward from S(T). If the two waves meet in the middle in identical
    states, causality is symmetric in this universe.
    """
    rng = np.random.default_rng(seed)
    S0 = (rng.random((size, size)) < 0.2).astype(np.uint8)

    forward_states = [S0.copy()]
    g = S0.copy()
    for t in range(steps):
        g = step_margolus(g, phase=t % 2)
        forward_states.append(g.copy())
    ST = forward_states[-1]

    backward_states = [ST.copy()]
    b = ST.copy()
    for t in reversed(range(steps)):
        b = step_margolus(b, phase=t % 2, reverse=True)
        backward_states.append(b.copy())

    mid = steps // 2
    return {
        "endpoint_recovered": bool(np.array_equal(backward_states[-1], S0)),
        "midpoint_match": bool(np.array_equal(forward_states[mid],
                                              backward_states[steps - mid])),
        "hamming_weight": int(S0.sum()),
    }
