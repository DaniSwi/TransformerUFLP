"""Estado del UFLP: el conjunto de instalaciones abiertas y sus cachés."""
from __future__ import annotations

import numpy as np

from .instance import UFLP_Instance


class UFLP_State:
    """Una solución (completa o parcial): el subconjunto S de abiertas.

    Como el UFLP no tiene restricciones de capacidad, fijado S la asignación de
    clientes es inmediata. Por eso S describe por completo una solución.

    Cachés
    ------
    assign[j]      índice de la instalación que atiende al cliente j
    best_cost[j]   costo de servir a j desde la abierta más cercana
    second_cost[j] costo desde la SEGUNDA más cercana

    El segundo caché es el truco de Whitaker (1983): al cerrar una instalación,
    cada cliente suyo se va exactamente a su segunda mejor opción, así que
    evaluar un `drop` no requiere recorrer S.
    """

    __slots__ = ("inst_info", "opened", "not_opened", "assign",
                 "best_cost", "second_cost", "cost")

    def __init__(self, inst_info: UFLP_Instance, opened=None):
        self.inst_info = inst_info
        M, N = inst_info.num_facilities, inst_info.num_clients

        self.opened = set(opened) if opened else set()
        self.not_opened = set(range(M)) - self.opened
        self.assign = np.full(N, -1, dtype=np.int64)
        self.best_cost = np.full(N, np.inf)
        self.second_cost = np.full(N, np.inf)
        self.cost = float("inf")        # S = {} es infactible

        if self.opened:
            self._recompute()

    # ------------------------------------------------------------------ cachés
    def _recompute(self):
        """Recalcula assign, best_cost y second_cost. O(|S| * N)."""
        c = self.inst_info.cost_matrix
        idx = np.fromiter(self.opened, dtype=np.int64, count=len(self.opened))
        sub = c[idx]

        if len(idx) == 1:
            self.assign = np.full(sub.shape[1], idx[0])
            self.best_cost = sub[0].copy()
            self.second_cost = np.full(sub.shape[1], np.inf)
        else:
            order = np.argpartition(sub, 1, axis=0)[:2]
            cols = np.arange(sub.shape[1])
            v0, v1 = sub[order[0], cols], sub[order[1], cols]
            swap = v0 > v1
            self.assign = idx[np.where(swap, order[1], order[0])]
            self.best_cost = np.where(swap, v1, v0)
            self.second_cost = np.where(swap, v0, v1)

        self.cost = float(self.inst_info.fixed_costs[idx].sum()
                          + self.best_cost.sum())

    # ------------------------------------------------------------------ utils
    def copy(self) -> "UFLP_State":
        s = UFLP_State.__new__(UFLP_State)
        s.inst_info = self.inst_info
        s.opened = set(self.opened)
        s.not_opened = set(self.not_opened)
        s.assign = self.assign.copy()
        s.best_cost = self.best_cost.copy()
        s.second_cost = self.second_cost.copy()
        s.cost = self.cost
        return s

    @property
    def size(self) -> int:
        return len(self.opened)

    def __repr__(self):
        return f"UFLP_State(|S|={len(self.opened)}, Z={self.cost:.3f})"
