<div align="center">

```
 ╔══════════════════════════════════════════════════╗
 ║  ┌───┐   TIME ONLY MOVES FORWARD                 ║
 ║  │▓▓▓│◄─ FOR EVERYONE ELSE.                        ║
 ║  └───┘   OUR CELLS CAN WALK BACK.                 ║
 ╚══════════════════════════════════════════════════╝
```

# timelock-ca

**Reversible, non-float compute substrates.** Four laboratories where
information is never destroyed: a topological charge lock, a bitwise
lattice gas with emergent gravity, a bit-exact time-travel cellular
automaton, and Hamiltonian neural networks trained with O(1) memory.

[![CI](https://github.com/kostyk348/timelock-ca/actions/workflows/ci.yml/badge.svg)](https://github.com/kostyk348/timelock-ca/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-green)]()

*README на русском: [README.ru.md](README.ru.md)*

</div>

---

## TL;DR

Modern computing is a one-way street: `x = f(x)` destroys the old `x`,
backprop stores every activation, floats round away your bits. This
project is four counterspies against entropy:

```console
$ timelock-ca margolus
{
  "entropy_conserved": true,
  "bit_exact_reversal": true,        ← 100 ticks forward, 100 back,
  "endpoint_recovered": true,          the lattice lands bit-for-bit
  "midpoint_match": true,              on its starting state
  "hamming_weight": 51
}

$ timelock-ca train
{
  "loss_first": 0.3313,
  "loss_last": 0.0848,
  "loss_decreased": true,
  "xor_accuracy_pct": 94.75          ← trained with ZERO stored activations
}
```

## Lab 1 — Topological charge lock (`topo`)

The Euler characteristic `χ = V − E + F` of a binary lattice, computed
with pure integer bit tricks, is treated as a **conserved charge**.
Random single-bit flips are proposed; each flip is *rejected* if it would
change χ. Two rings (χ = 0 each) under 200 noisy proposals: the mesh
wriggles, the topology is locked.

```
before: two rings            after 200 flips: still two rings
┌──────────┐                 ┌──────────┐
│ ┌──────┐ │   χ = 0         │ ┌──────┐ │   χ = 0
│ │ hole │ │   ─────►        │ │ hole │ │   ─────►  (jagged edges,
│ └──────┘ │                 │ └──────┘ │    same topology)
└──────────┘                 └──────────┘
```

This is "topological protection" — the same principle as anyon-based
quantum memory — computed in pure `uint8` numpy.

## Lab 2 — Bitwise lattice-gas gravity (`hpp`)

HPP lattice gas: 4 boolean channels (Up/Down/Left/Right) stream, bounce
off two obstacle disks, collide head-on. Measure the pressure imbalance
on each disk: a net **attractive force** emerges — gravity from pure
bits.

Gotcha fixed in production: the naive sequential bounce-back
(`grid[opp] |= bounced`) silently *annihilates* particles. The
simultaneous pairwise swap (`d ↔ opposite` inside obstacles, single
snapshot) is strictly mass-conserving, and the test suite proves it.

## Lab 3 — Bit-exact time travel (`margolus`)

The Billiard Ball Model as a Margolus partitioned CA. Each 2×2 block
transforms by a **permutation** of the 16 possible states (a LUT):

```
0001 → 0010 → 0100 → 1000 → 0001     (single particle orbits: momentum kept)
1001 ↔ 0110                           (elastic two-particle collision)
```

Because every transition is a bijection:
- Hamming weight (particle count) is *strictly invariant* — `dH/dt = 0`;
- time reversal is the inverse LUT — no recomputation, no checkpointing;
- **meet-in-the-middle**: run T steps forward from S₀ and T steps backward
  from S(T); the two time-waves meet in identical states. Causality is
  symmetric here.

Our BBM table is verified 16-element bijective by tests, and
`forward∘backward == identity` bit-for-bit at 100 steps.

## Lab 4 — Symplectic networks with O(1) training memory (`train`)

A neural network as a *Hamiltonian system*: coordinates `q` and momenta
`p` evolve under leapfrog steps,

```
p_{i+1} = p_i + dt·tanh(q_i @ W_i + b_i)
q_{i+1} = q_i + dt·p_{i+1}
```

Every step is exactly invertible, so backprop does not need stored
activations: an **adjoint wave** walks time backwards, recomputing each
layer on the fly. Peak activation memory: **strictly 0 bytes**, for any
depth. Plus Adam, full-batch, gradients derived by hand.

| experiment                          | result (this repo's defaults)      |
|-------------------------------------|------------------------------------|
| XOR separation, dim=4, depth=12     | 94.75% accuracy in 120 epochs      |
| sklearn digits (64-D phase space)   | **97.1%** test accuracy in 60 epochs, full batch |

The digits model has a linear readout head on the phase-space boundary;
cross-entropy gradients ride the adjoint wave back through 8 leapfrog
layers.

## Install

```bash
pip install -e .            # core (numpy)
pip install -e .[ml]        # + scikit-learn for the digits experiment
```

## CLI reference

```
timelock-ca topo      [--size 32] [--steps 200]
timelock-ca hpp       [--size 64] [--steps 300]
timelock-ca margolus  [--size 16] [--steps 100] [--mid-steps 40]
timelock-ca train     [--epochs 120] [--depth 12] [--n 400]
```

All commands emit JSON — pipe them into `jq` (why not the Windows one) and
wire them into CI gates: `timelock-ca margolus` exits non-zero if time
travel stops being bit-exact. Quality assurance for causality.

## Python API

```python
from timelock_ca import (HPPLattice, TopologicalLattice,
                         step_margolus, verify_reversibility,
                         ExactSymplecticNet, SymplecticClassifier)

# time travel
report = verify_reversibility(size=16, steps=100)
assert report["bit_exact_reversal"]

# custom reversible rule: any permutation of 0..15 is a valid universe
```

## Repository layout

```
timelock-ca/
├── src/timelock_ca/
│   ├── topology.py    euler_characteristic, TopologicalLattice
│   ├── hpp.py         HPPLattice: stream/bounce/collide/measure
│   ├── margolus.py    BBM LUTs, forward/backward, meet-in-the-middle
│   ├── symplectic.py  leapfrog flows, exact adjoint + Adam, digits classifier
│   └── cli.py         four JSON-reporting labs
├── tests/             16 pytest cases: bijections, conservation, bit-exactness
├── examples/all_labs.py
├── notebooks/         the original research notebook (Colab)
└── .github/workflows/ci.yml
```

## Why reversible computing matters

- **Landauer's principle**: erasing a bit costs `kT·ln2` in heat. A
  bit-exact reversible computer sidesteps the fundamental floor.
- **Adjoint memory**: reversible networks make deep models trainable on
  tiny VRAM (this is the same trick as RevNets/gradient checkpointing,
  taken to its exact conclusion).
- **Physics simulation**: lattice gases *are* Navier–Stokes in the limit,
  computed without a single floating-point multiply.
- **It's beautiful**: a universe where you can always undo.

## Related work

- Margolus, Toffoli — *Cellular Automata Machines* (the BBM tables live here).
- Frisch, Hasslacher, Pomeau — lattice-gas hydrodynamics.
- Chen et al., *The Reversible Residual Network* (RevNet) — backprop without
  stored activations, the industrial cousin of Lab 4.
- Gomez et al., *Reversible Recurrent Neural Networks*.

## License

Apache-2.0 — see [LICENSE](LICENSE).
