"""Agentes y algoritmos tradicionales: Greedy Add, Hill Climbing, ILS.

Siguen la arquitectura del módulo 1: un agente es una política que, dado un
estado, devuelve una acción (o None para terminar); el solver aplica la política
en bucle.
"""
from __future__ import annotations

import random

import numpy as np

from .environment import UFLP_Environment as env
from .instance import UFLP_Instance
from .state import UFLP_State


# ============================================================ evaluadores
def evalConstructiveActions(state: UFLP_State, _env=env):
    """Heurística Greedy Add: puntaje = ahorro neto -ΔZ (mayor es mejor).

    Solo reporta acciones con mejora estricta; al no quedar ninguna el agente
    recibe una lista vacía y detiene la construcción.
    """
    evals = []
    for a in _env.gen_actions(state, "constructive"):
        new_z = _env.calculate_cost_after_action(state, a)
        if new_z < state.cost:
            evals.append((a, state.cost - new_z))
    return evals


# ================================================================ agentes
class GreedyAgent:
    """Elige la acción de mayor evaluación."""

    def __init__(self, eval_actions):
        self.eval_actions = eval_actions

    def reset(self):
        pass

    def action_policy(self, state, environment=env):
        evals = self.eval_actions(state, environment)
        if not evals:
            return None
        return max(evals, key=lambda x: x[1])[0]


class LocalSearchAgent:
    """Hill Climbing sobre el vecindario unificado Add u Drop u Swap."""

    def __init__(self, action_type: str = "local-search",
                 first_improvement: bool = False):
        self.action_type = action_type
        self.first_improvement = first_improvement

    def reset(self):
        pass

    def action_policy(self, state, environment=env):
        best_a, best_gain = None, 0.0
        for a in environment.gen_actions(state, self.action_type,
                                         shuffle=self.first_improvement):
            gain = state.cost - environment.calculate_cost_after_action(state, a)
            if gain > 1e-9:
                if self.first_improvement:
                    return a
                if gain > best_gain:
                    best_a, best_gain = a, gain
        return best_a


class SingleAgentSolver:
    """Aplica la política del agente hasta que devuelva None."""

    def __init__(self, environment, agent):
        self.env = environment
        self.agent = agent

    def solve(self, state: UFLP_State, max_actions: int = 0):
        if max_actions == 0:
            max_actions = 10 ** 9
        self.agent.reset()
        n = 0
        while n < max_actions:
            a = self.agent.action_policy(state, self.env)
            if a is None:
                break
            state = self.env.state_transition(state, a)
            n += 1
        return state, None, n


# ======================================================= atajos funcionales
def greedy_add(inst: UFLP_Instance) -> UFLP_State:
    """Constructivo voraz. Es la heurística clásica de referencia."""
    return SingleAgentSolver(env, GreedyAgent(evalConstructiveActions)).solve(
        UFLP_State(inst))[0]


def hill_climbing(state: UFLP_State, first_improvement: bool = False) -> UFLP_State:
    """Búsqueda local hasta un óptimo local."""
    return SingleAgentSolver(env, LocalSearchAgent(
        first_improvement=first_improvement)).solve(state)[0]


def random_constructive(inst: UFLP_Instance,
                        rng: random.Random | None = None) -> UFLP_State:
    """Baseline aleatorio: en cada paso abre una mejoradora al azar.

    Es el "solver constructivo aleatorio" contra el que la rúbrica pide
    comparar como piso mínimo.
    """
    rng = random.Random(0) if rng is None else rng
    state = UFLP_State(inst)
    while True:
        mejoradoras = [a for a in env.gen_actions(state, "constructive")
                       if env.calculate_cost_after_action(state, a) < state.cost]
        if not mejoradoras:
            return state
        state = env.state_transition(state, rng.choice(mejoradoras))


class ILS:
    """Iterated Local Search: perturbación -> hill climbing -> aceptación."""

    def __init__(self, environment=env, k: int = 2, max_iterations: int = 25,
                 seed: int | None = None):
        self.env = environment
        self.k = k
        self.max_iterations = max_iterations
        self.rng = random.Random(seed)

    def solve(self, state: UFLP_State) -> UFLP_State:
        best = hill_climbing(state)
        for _ in range(self.max_iterations):
            pert = best
            for _ in range(self.k):                            # 1. perturbación
                acts = list(self.env.gen_actions(pert, "swap"))
                if not acts:
                    break
                pert = self.env.state_transition(pert, self.rng.choice(acts))
            cand = hill_climbing(pert)                         # 2. refinamiento
            if cand.cost < best.cost - 1e-9:                   # 3. aceptación
                best = cand
        return best
