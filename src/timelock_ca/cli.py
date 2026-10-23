"""timelock-ca CLI — four laboratories of reversible compute."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .hpp import HPPLattice
from .margolus import meet_in_the_middle, verify_reversibility
from .symplectic import ExactSymplecticNet
from .topology import TopologicalLattice


def cmd_topo(args) -> int:
    lat = TopologicalLattice(size=args.size, seed=args.seed)
    lat.stamp_ring(6, 6)
    lat.stamp_ring(18, 18)
    chi0 = lat.chi
    accepted = lat.diffuse(args.steps)
    print(json.dumps({
        "chi_initial": chi0,
        "chi_final": lat.chi,
        "topology_locked": lat.chi == chi0,
        "accepted_flips": accepted,
        "rejected_flips": lat.rejected_flips,
    }, indent=2))
    return 0


def cmd_hpp(args) -> int:
    gas = HPPLattice(size=args.size, seed=args.seed)
    gas.run(args.steps)
    mean_f = gas.mean_force
    print(json.dumps({
        "steps": args.steps,
        "mean_net_force": round(mean_f, 4),
        "attraction_detected": bool(mean_f > 0),
        "final_activity": float(gas.grid.sum()),
    }, indent=2))
    return 0


def cmd_margolus(args) -> int:
    report = verify_reversibility(size=args.size, steps=args.steps, seed=args.seed)
    mid = meet_in_the_middle(size=args.size, steps=args.mid_steps, seed=args.seed)
    print(json.dumps({**report, **mid}, indent=2))
    return 0 if report["bit_exact_reversal"] else 1


def cmd_train(args) -> int:
    rng = np.random.default_rng(args.seed)
    n = args.n
    X2 = rng.uniform(-1.0, 1.0, (n, 2))
    y = ((X2[:, 0] > 0) ^ (X2[:, 1] > 0)).astype(int)
    X4 = np.zeros((n, 4))
    X4[:, 0], X4[:, 1] = X2[:, 0], X2[:, 1]

    model = ExactSymplecticNet(depth=args.depth, dt=0.15, dim=4, seed=args.seed)
    losses = []
    for _epoch in range(args.epochs):
        losses.append(model.train_step(X4, y, lr=0.03))
    acc = model.accuracy(X4, y)
    print(json.dumps({
        "epochs": args.epochs,
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "loss_decreased": bool(losses[-1] < losses[0]),
        "xor_accuracy_pct": round(acc * 100, 2),
    }, indent=2))
    return 0 if acc > 0.9 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="timelock-ca",
                                     description="Reversible / non-float compute labs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("topo", help="Euler-characteristic topological lock")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_topo)

    p = sub.add_parser("hpp", help="bitwise lattice-gas attraction")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_hpp)

    p = sub.add_parser("margolus", help="bit-exact time travel")
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--mid-steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_margolus)

    p = sub.add_parser("train", help="adjoint-trained symplectic XOR separator")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_train)

    ns = parser.parse_args(argv)
    return ns.fn(ns)


if __name__ == "__main__":
    raise SystemExit(main())
