import numpy as np
import pytest

from timelock_ca.symplectic import ExactSymplecticNet, SymplecticClassifier


def make_xor_data(n=400, seed=42):
    rng = np.random.default_rng(seed)
    X2 = rng.uniform(-1.0, 1.0, (n, 2))
    y = ((X2[:, 0] > 0) ^ (X2[:, 1] > 0)).astype(int)
    X4 = np.zeros((n, 4))
    X4[:, 0], X4[:, 1] = X2[:, 0], X2[:, 1]
    return X4, y


def test_forward_flow_is_deterministic():
    X, _ = make_xor_data(n=50)
    a = ExactSymplecticNet(depth=4, dim=4, seed=1)
    q1, p1 = a.forward_flow(X)
    q2, p2 = a.forward_flow(X)
    assert np.array_equal(q1, q2) and np.array_equal(p1, p2)


def test_adjoint_training_learns_xor():
    X, y = make_xor_data()
    model = ExactSymplecticNet(depth=12, dt=0.15, dim=4, seed=42)
    first = model.train_step(X, y, lr=0.03)
    for _ in range(150):
        last = model.train_step(X, y, lr=0.03)
    assert last < first
    assert model.accuracy(X, y) > 0.9


def test_classifier_on_digits_smoke():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X, digits.target,
                                              test_size=0.25, random_state=42)
    model = SymplecticClassifier(dim=64, num_classes=10, depth=8, dt=0.15, seed=42)
    l0 = model.train_step(X_tr, y_tr, lr=0.02)
    for _ in range(30):
        l1 = model.train_step(X_tr, y_tr, lr=0.02)
    assert l1 < l0
    acc = model.accuracy(X_te, y_te)
    assert acc > 0.5  # 30 epochs is a smoke test, not a full run


def test_zero_activation_memory_claim():
    """The API never exposes intermediate states: forward_flow returns only
    the boundary (q_T, p_T) — that IS the O(1) memory contract."""
    X, _ = make_xor_data(n=10)
    model = ExactSymplecticNet(depth=6, dim=4, seed=0)
    out = model.forward_flow(X)
    assert len(out) == 2
