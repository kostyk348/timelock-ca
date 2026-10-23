import numpy as np

from timelock_ca.hpp import HPPLattice


def test_mass_conserved_under_stream_and_bounce():
    gas = HPPLattice(size=32, seed=0)
    m0 = int(gas.grid.sum())
    for _ in range(20):
        gas.step()
    m1 = int(gas.grid.sum())
    assert m0 == m1  # streaming + bounce-back + rotation conserve particles


def test_force_history_grows():
    gas = HPPLattice(size=64, seed=42)
    gas.run(300)
    assert len(gas.force_history) == 300
    # emergent attraction: positive mean net force in the notebook setup
    assert gas.mean_force > 0


def test_no_uint8_underflow():
    gas = HPPLattice(size=32, seed=3)
    gas.run(50)
    # all channels stay strictly binary
    assert set(np.unique(gas.grid)).issubset({0, 1})
