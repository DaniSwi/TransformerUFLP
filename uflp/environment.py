"""Reglas del mundo: acciones, deltas de costo en O(N) y transiciones."""
from __future__ import annotations

import random

import numpy as np

from .state import UFLP_State

Action = tuple[str, object]


class UFLP_Environment:
    """Ambiente estático y desacoplado del estado.

    Acciones
    --------
    ("constructive", i)   abrir i  (alias "add")
    ("drop", i)           cerrar i, solo si |S| > 1
    ("swap", (i1, i2))    cerrar i1 y abrir i2

    El vecindario unificado Add u Drop u Swap es el que da la cota de
    aproximación de factor 3 de Arya et al. (2001).
    """

    # ------------------------------------------------------------- acciones
    @staticmethod
    def gen_actions(state: UFLP_State, action_type: str = "constructive",
                    shuffle: bool = False):
        if action_type in ("constructive", "add"):
            kind = "constructive" if action_type == "constructive" else "add"
            cand = list(state.not_opened)
            if shuffle:
                random.shuffle(cand)
            for i in cand:
                yield (kind, i)

        elif action_type == "drop":
            if len(state.opened) > 1:              # |S| > 1 evita el estado vacío
                cand = list(state.opened)
                if shuffle:
                    random.shuffle(cand)
                for i in cand:
                    yield ("drop", i)

        elif action_type == "swap":
            pairs = [(i1, i2) for i1 in state.opened for i2 in state.not_opened]
            if shuffle:
                random.shuffle(pairs)
            for p in pairs:
                yield ("swap", p)

        elif action_type in ("local-search", None):
            acts = (list(UFLP_Environment.gen_actions(state, "add"))
                    + list(UFLP_Environment.gen_actions(state, "drop"))
                    + list(UFLP_Environment.gen_actions(state, "swap")))
            if shuffle:
                random.shuffle(acts)
            for a in acts:
                yield a

        else:
            raise ValueError(f"tipo de acción desconocido: {action_type}")

    # --------------------------------------------------------------- deltas
    @staticmethod
    def calculate_cost_after_action(state: UFLP_State, action: Action) -> float:
        """Nuevo Z(S') en O(N), sin instanciar el estado resultante."""
        kind, arg = action
        c = state.inst_info.cost_matrix
        f = state.inst_info.fixed_costs

        if kind in ("constructive", "add"):
            i = arg
            if not state.opened:                       # S = {} -> Z0(i)
                return float(f[i] + c[i].sum())
            gain = np.maximum(0.0, state.best_cost - c[i]).sum()
            return float(state.cost + f[i] - gain)

        if kind == "drop":                             # truco de Whitaker
            i = arg
            mask = state.assign == i
            return float(state.cost - f[i]
                         + (state.second_cost[mask] - state.best_cost[mask]).sum())

        if kind == "swap":
            i1, i2 = arg
            mask = state.assign == i1
            new = np.where(mask,
                           np.minimum(state.second_cost, c[i2]),   # clientes de i1
                           np.minimum(state.best_cost, c[i2]))     # el resto
            return float(f[i2] - f[i1] + f[list(state.opened)].sum() + new.sum())

        raise ValueError(f"acción desconocida: {action}")

    # ---------------------------------------------------------- transición
    @staticmethod
    def state_transition(state: UFLP_State, action: Action) -> UFLP_State:
        """Aplica la acción sobre una copia del estado y devuelve el nuevo."""
        kind, arg = action
        s = state.copy()

        if kind in ("constructive", "add"):
            s.opened.add(arg)
            s.not_opened.discard(arg)
        elif kind == "drop":
            s.opened.discard(arg)
            s.not_opened.add(arg)
        elif kind == "swap":
            i1, i2 = arg
            s.opened.discard(i1)
            s.not_opened.add(i1)
            s.opened.add(i2)
            s.not_opened.discard(i2)
        else:
            raise ValueError(f"acción desconocida: {action}")

        s._recompute()
        return s


env = UFLP_Environment
