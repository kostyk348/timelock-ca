"""Bitwise HPP lattice gas: attraction between two bodies from pure bits.

Four boolean channels (Up/Down/Left/Right) stream across a lattice, bounce
off two obstacle disks and collide head-on. Measuring the pressure
imbalances on each disk reveals a net *attractive* force between them —
gravity emerging from a reversible bit substrate, no floats involved
(until you plot the force history).
"""

from __future__ import annotations

import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class HPPLattice:
    def __init__(self, size: int = 64, seed: int = 42,
                 body1_x: int | None = None, body2_x: int | None = None,
                 body_y: int | None = None, body_r: int = 3) -> None:
        self.size = size
        rng = np.random.default_rng(seed)
        # 4 direction channels, ~25% occupancy
        self.grid = (rng.random((4, size, size)) < 0.25).astype(np.uint8)

        # scale default geometry with the lattice so any size works
        body1_x = int(size * 0.375) if body1_x is None else body1_x
        body2_x = int(size * 0.625) if body2_x is None else body2_x
        body_y = size // 2 if body_y is None else body_y
        if not (body1_x - body_r - 1 >= 0 and body2_x + body_r + 1 < size):
            raise ValueError("bodies too close to the lattice edge for pressure probes")

        yy, xx = np.ogrid[:size, :size]
        self.obstacle_mask = (
            ((xx - body1_x) ** 2 + (yy - body_y) ** 2 <= body_r ** 2) |
            ((xx - body2_x) ** 2 + (yy - body_y) ** 2 <= body_r ** 2)
        )
        self.body1_x, self.body2_x, self.body_y, self.body_r = body1_x, body2_x, body_y, body_r
        self.force_history: list[float] = []
        self.density_accum = np.zeros((size, size), dtype=np.float64)

    # ------------------------------------------------------------------
    def _stream(self) -> None:
        g = self.grid
        g[UP] = np.roll(g[UP], -1, axis=0)
        g[DOWN] = np.roll(g[DOWN], 1, axis=0)
        g[LEFT] = np.roll(g[LEFT], -1, axis=1)
        g[RIGHT] = np.roll(g[RIGHT], 1, axis=1)

    def _bounce_back(self) -> None:
        """Simultaneous bounce-back: inside obstacles, d <-> opposite swap.

        Doing the swap pairwise-simultaneously (single snapshot) is what makes
        this strictly mass-conserving — the naive sequential version silently
        annihilates particles via `|=`.
        """
        g = self.grid
        snap = g.copy()
        m = self.obstacle_mask
        for d, opp in ((UP, DOWN), (LEFT, RIGHT)):
            g[d] = (snap[d] & ~m) | (snap[opp] & m)
            g[opp] = (snap[opp] & ~m) | (snap[d] & m)

    def _collide(self) -> None:
        """HPP head-on collision: rotate head-on pairs by 90 degrees.

        The rotation is its own inverse and conserves mass/momentum exactly,
        which is what makes HPP a legitimate *reversible* physics substrate.
        """
        g = self.grid
        head_on_v = (g[UP] == 1) & (g[DOWN] == 1) & (g[LEFT] == 0) & (g[RIGHT] == 0)
        head_on_h = (g[LEFT] == 1) & (g[RIGHT] == 1) & (g[UP] == 0) & (g[DOWN] == 0)
        rotate = head_on_v | head_on_h

        up_old = g[UP].copy()
        down_old = g[DOWN].copy()
        left_old = g[LEFT].copy()
        right_old = g[RIGHT].copy()
        g[UP][rotate] = left_old[rotate]
        g[LEFT][rotate] = up_old[rotate]
        g[DOWN][rotate] = right_old[rotate]
        g[RIGHT][rotate] = down_old[rotate]

    def _measure_force(self) -> float:
        """Net pressure imbalance on both disks -> emergent attraction."""
        g = self.grid
        y0, y1 = self.body_y - self.body_r, self.body_y + self.body_r + 1
        b1, b2, r = self.body1_x, self.body2_x, self.body_r

        # explicit int() everywhere: no uint8 underflow games
        p_left_1 = int(np.sum(g[LEFT, y0:y1, b1 - r - 1]))
        p_right_1 = int(np.sum(g[RIGHT, y0:y1, b1 + r + 1]))
        force_1 = p_left_1 - p_right_1

        p_left_2 = int(np.sum(g[LEFT, y0:y1, b2 - r - 1]))
        p_right_2 = int(np.sum(g[RIGHT, y0:y1, b2 + r + 1]))
        force_2 = p_right_2 - p_left_2

        return (force_1 + force_2) / 2.0

    # ------------------------------------------------------------------
    def step(self) -> float:
        self._stream()
        self._bounce_back()
        self.density_accum += self.grid.sum(axis=0, dtype=np.int64)
        force = self._measure_force()
        self.force_history.append(force)
        self._collide()
        return force

    def run(self, steps: int = 300) -> list[float]:
        for _ in range(steps):
            self.step()
        return self.force_history

    @property
    def mean_force(self) -> float:
        return float(np.mean(self.force_history)) if self.force_history else 0.0
