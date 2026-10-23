"""Topological charge: the Euler characteristic as a conserved quantity.

chi = V - E + F computed with pure integer bit tricks on a binary lattice:

* V — number of active cells
* E — number of adjacent horizontal/vertical active pairs
* F — number of fully active 2x2 plaquettes

Random boundary noise is *rejected* whenever it would change chi: the
deformation of the mesh is free, the topology is locked. Two rings with a
hole each keep chi stable under thousands of proposed flips.
"""

from __future__ import annotations

import numpy as np


def euler_characteristic(grid: np.ndarray) -> int:
    """chi = V - E + F for a binary (uint8) grid."""
    g = grid.astype(np.uint8, copy=False)
    V = int(np.sum(g))
    E_h = int(np.sum(g[:, :-1] & g[:, 1:]))
    E_v = int(np.sum(g[:-1, :] & g[1:, :]))
    E = E_h + E_v
    F = int(np.sum(g[:-1, :-1] & g[1:, :-1] & g[:-1, 1:] & g[1:, 1:]))
    return V - E + F


class TopologicalLattice:
    """A lattice whose topology is a conserved quantity.

    ``flip`` proposes single-bit flips; they are accepted only if the Euler
    characteristic survives. The result is a system that can wriggle but
    never tear: an integer-locked 'topological charge'.
    """

    def __init__(self, size: int = 32, seed: int | None = None) -> None:
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.chi = 0
        self.rejected_flips = 0

    def stamp_ring(self, y0: int, x0: int, outer: int = 8, hole: int = 4) -> None:
        """Stamp one ring (filled square with a centered square hole)."""
        self.grid[y0:y0 + outer, x0:x0 + outer] = 1
        o = (outer - hole) // 2
        self.grid[y0 + o:y0 + o + hole, x0 + o:x0 + o + hole] = 0
        self.recompute()

    def recompute(self) -> None:
        self.chi = euler_characteristic(self.grid)

    def flip(self, y: int | None = None, x: int | None = None) -> bool:
        """Propose one bit flip; keep it only if chi is unchanged."""
        if y is None or x is None:
            y, x = self.rng.integers(1, self.size - 1, size=2)
        before = self.grid[y, x]
        self.grid[y, x] ^= 1
        if euler_characteristic(self.grid) != self.chi:
            self.grid[y, x] = before  # revert: topological charge locked!
            self.rejected_flips += 1
            return False
        return True

    def diffuse(self, steps: int = 50) -> int:
        """Run ``steps`` proposed flips, return accepted count."""
        accepted = 0
        for _ in range(steps):
            accepted += self.flip()
        return accepted
