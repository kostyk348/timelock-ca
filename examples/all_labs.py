#!/usr/bin/env python3
"""One-page tour of all four reversible-compute laboratories."""

import numpy as np

from timelock_ca.hpp import HPPLattice
from timelock_ca.margolus import meet_in_the_middle, verify_reversibility
from timelock_ca.symplectic import ExactSymplecticNet
from timelock_ca.topology import TopologicalLattice

# 1. topological charge lock
lat = TopologicalLattice(size=32, seed=7)
lat.stamp_ring(6, 6)
lat.stamp_ring(18, 18)
accepted = lat.diffuse(200)
print(f"[topology ] chi={lat.chi} locked, {accepted} flips accepted, "
      f"{lat.rejected_flips} rejected")

# 2. lattice-gas gravity
gas = HPPLattice(size=64, seed=42)
gas.run(300)
print(f"[hpp      ] mean net force between bodies = {gas.mean_force:+.3f}")

# 3. bit-exact time travel
report = verify_reversibility(size=16, steps=100, seed=0)
print(f"[margolus ] entropy kept: {report['entropy_conserved']}, "
      f"bit-exact reversal: {report['bit_exact_reversal']}")
mid = meet_in_the_middle(size=32, steps=40, seed=1337)
print(f"[margolus ] meet-in-the-middle endpoint recovered: {mid['endpoint_recovered']}")

# 4. symplectic adjoint training (O(1) activation memory)
rng = np.random.default_rng(42)
X2 = rng.uniform(-1, 1, (400, 2))
y = ((X2[:, 0] > 0) ^ (X2[:, 1] > 0)).astype(int)
X4 = np.zeros((400, 4))
X4[:, 0], X4[:, 1] = X2[:, 0], X2[:, 1]
model = ExactSymplecticNet(depth=12, dt=0.15, dim=4, seed=42)
for _ in range(120):
    model.train_step(X4, y, lr=0.03)
print(f"[symplectic] XOR accuracy after 120 epochs: {model.accuracy(X4, y) * 100:.1f}%")
