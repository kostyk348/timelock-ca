"""Symplectic (Hamiltonian) neural flows with O(1)-memory training.

A network is a *physical system*: coordinates q evolve under leapfrog steps
with conserved phase-space volume. Training does not need stored
activations — the leapfrog step is exactly invertible, so gradients are
computed by an adjoint wave that walks time backwards, recomputing each
layer on the fly. Peak memory for activations: strictly 0 bytes.
"""

from __future__ import annotations

import numpy as np


class _Adam:
    """Tiny Adam over groups of (params, grads, ms, vs). One t-tick per call."""

    def __init__(self) -> None:
        self.t = 0

    def update(self, groups, lr: float) -> None:
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for params, grads, ms, vs in groups:
            for p, g, m, v in zip(params, grads, ms, vs):
                m[:] = beta1 * m + (1 - beta1) * g
                v[:] = beta2 * v + (1 - beta2) * (g * g)
                m_hat = m / (1 - beta1 ** self.t)
                v_hat = v / (1 - beta2 ** self.t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)


class SymplecticFlowBase:
    """Shared leapfrog machinery: forward flow and its exact inverse."""

    def __init__(self, dim: int, depth: int, dt: float = 0.15, seed: int = 42) -> None:
        self.dim, self.depth, self.dt = dim, depth, dt
        rng = np.random.default_rng(seed)
        self.W = [rng.standard_normal((dim, dim)) / np.sqrt(dim) for _ in range(depth)]
        self.b = [np.zeros(dim) for _ in range(depth)]
        self._opt = _Adam()
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]

    def _force(self, q: np.ndarray, i: int) -> np.ndarray:
        return np.tanh(q @ self.W[i] + self.b[i])

    def forward_flow(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """t: 0 -> T. Returns (q_T, p_T). Nothing is stored."""
        q = X.copy()
        p = np.zeros_like(q)
        for i in range(self.depth):
            p = p + self.dt * self._force(q, i)
            q = q + self.dt * p
        return q, p


class ExactSymplecticNet(SymplecticFlowBase):
    """XOR-style phase separator trained by the exact adjoint method.

    Loss: push class 0 orbits to radius ~0.4, class 1 orbits to radius ~1.6.
    """

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.03) -> float:
        n = X.shape[0]
        q_end, p_end = self.forward_flow(X)

        r = np.linalg.norm(q_end, axis=1, keepdims=True)
        target_r = np.where(np.asarray(y).reshape(-1, 1) == 1, 1.6, 0.4)
        loss = float(np.mean((r - target_r) ** 2))

        # adjoint boundary conditions at t = T
        g_q = 2.0 * (r - target_r) * (q_end / (r + 1e-7)) / n
        g_p = np.zeros_like(g_q)

        q, p = q_end.copy(), p_end.copy()
        dW_all = [None] * self.depth
        db_all = [None] * self.depth

        for i in reversed(range(self.depth)):
            q_prev = q - self.dt * p              # exact leapfrog inversion
            g_p = g_p + self.dt * g_q             # adjoint momentum transport
            act = np.tanh(q_prev @ self.W[i] + self.b[i])
            du = (g_p * self.dt) * (1.0 - act ** 2)
            dW_all[i] = q_prev.T @ du
            db_all[i] = np.sum(du, axis=0)
            g_q = g_q + du @ self.W[i].T          # adjoint coordinate transport
            p = p - self.dt * act                 # exact momentum inversion
            q = q_prev

        self._opt.update(
            [(self.W, dW_all, self._mW, self._vW),
             (self.b, db_all, self._mb, self._vb)], lr)
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        q_end, _ = self.forward_flow(X)
        r = np.linalg.norm(q_end, axis=1)
        return (r > 1.0).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class SymplecticClassifier(SymplecticFlowBase):
    """Leapfrog body + linear readout head, cross-entropy loss.

    Trains on sklearn digits (8x8, 64D phase space) exactly like the
    notebook: full-batch Adam, adjoint backward pass, zero activation
    memory.
    """

    def __init__(self, dim: int, num_classes: int, depth: int = 8,
                 dt: float = 0.15, seed: int = 42) -> None:
        super().__init__(dim, depth, dt, seed=seed)
        rng = np.random.default_rng(seed)
        self.num_classes = num_classes
        self.W_head = rng.standard_normal((dim, num_classes)) * 0.1
        self.b_head = np.zeros(num_classes)
        self._mWh = np.zeros_like(self.W_head)
        self._vWh = np.zeros_like(self.W_head)
        self._mbh = np.zeros_like(self.b_head)
        self._vbh = np.zeros_like(self.b_head)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.02) -> float:
        n = X.shape[0]
        q_end, p_end = self.forward_flow(X)

        logits = q_end @ self.W_head + self.b_head
        probs = self._softmax(logits)
        loss = float(-np.mean(np.log(probs[np.arange(n), y] + 1e-12)))

        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        dW_head = q_end.T @ dlogits
        db_head = np.sum(dlogits, axis=0)

        g_q = dlogits @ self.W_head.T
        g_p = np.zeros_like(g_q)

        q, p = q_end.copy(), p_end.copy()
        dW_all = [None] * self.depth
        db_all = [None] * self.depth

        for i in reversed(range(self.depth)):
            q_prev = q - self.dt * p
            g_p = g_p + self.dt * g_q
            act = np.tanh(q_prev @ self.W[i] + self.b[i])
            du = (g_p * self.dt) * (1.0 - act ** 2)
            dW_all[i] = q_prev.T @ du
            db_all[i] = np.sum(du, axis=0)
            g_q = g_q + du @ self.W[i].T
            p = p - self.dt * act
            q = q_prev

        self._opt.update(
            [(self.W, dW_all, self._mW, self._vW),
             (self.b, db_all, self._mb, self._vb),
             ([self.W_head], [dW_head], [self._mWh], [self._vWh]),
             ([self.b_head], [db_head], [self._mbh], [self._vbh])], lr)
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        q_end, _ = self.forward_flow(X)
        return np.argmax(q_end @ self.W_head + self.b_head, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
