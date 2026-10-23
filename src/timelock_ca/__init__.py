"""timelock-ca — reversible, non-float compute substrates.

Four laboratories of computation that never lose information:

* :mod:`timelock_ca.topology`  — Euler characteristic as a locked topological charge
* :mod:`timelock_ca.hpp`       — bitwise HPP lattice gas with emergent attraction
* :mod:`timelock_ca.margolus`  — Margolus/BBM block cellular automaton, bit-exactly reversible
* :mod:`timelock_ca.symplectic`— Hamiltonian neural flows trained by exact adjoint (O(1) memory)
"""

from .topology import TopologicalLattice, euler_characteristic
from .hpp import HPPLattice
from .margolus import BBM_FWD, BBM_REV, step_margolus, verify_reversibility
from .symplectic import ExactSymplecticNet, SymplecticClassifier

__version__ = "0.1.0"

__all__ = [
    "TopologicalLattice", "euler_characteristic",
    "HPPLattice",
    "BBM_FWD", "BBM_REV", "step_margolus", "verify_reversibility",
    "ExactSymplecticNet", "SymplecticClassifier",
    "__version__",
]
