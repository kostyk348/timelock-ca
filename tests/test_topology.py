import numpy as np
import pytest

from timelock_ca.topology import TopologicalLattice, euler_characteristic


def test_euler_of_known_shapes():
    filled = np.ones((4, 4), dtype=np.uint8)
    # solid square: genus 0
    assert euler_characteristic(filled) == 1

    ring = np.ones((6, 6), dtype=np.uint8)
    ring[2:4, 2:4] = 0
    # one hole -> chi 0
    assert euler_characteristic(ring) == 0

    assert euler_characteristic(np.zeros((8, 8), dtype=np.uint8)) == 0


def test_two_rings_lock_topology():
    lat = TopologicalLattice(size=32, seed=7)
    lat.stamp_ring(6, 6)
    lat.stamp_ring(18, 18)
    chi0 = lat.chi
    accepted = lat.diffuse(200)
    assert lat.chi == chi0
    assert lat.rejected_flips > 0
    assert accepted >= 0


def test_flip_reverts_on_change():
    lat = TopologicalLattice(size=16, seed=1)
    lat.stamp_ring(4, 4)
    lat.grid[4, 4] = 0  # break the ring -> chi must change
    lat.recompute()
    chi_broken = lat.chi
    assert chi_broken != 1
