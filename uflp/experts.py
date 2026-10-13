"""El experto a imitar.

Cambiar de experto es cambiar un string: la descomposición y el resto del
pipeline son idénticos, solo cambia de dónde sale S*.

    "ils"     Greedy Add -> Hill Climbing -> ILS.  El del módulo 1.
    "exact"   óptimo probado con un solver MIP.
    "greedy"  solo Greedy Add. Sirve como control: imitarlo no debería
              dar nada mejor que el propio Greedy.
"""
from __future__ import annotations

from .agents import ILS, greedy_add, hill_climbing
from .config import DATA
from .exact import solve_exact
from .instance import UFLP_Instance
from .state import UFLP_State

METHODS = ("ils", "exact", "greedy")


def solve_expert(inst: UFLP_Instance, method: str | None = None,
                 seed: int = 0, ils_iterations: int | None = None,
                 ils_k: int | None = None) -> UFLP_State:
    """Resuelve una instancia con el experto elegido."""
    method = DATA.expert if method is None else method

    if method == "greedy":
        return greedy_add(inst)

    if method == "exact":
        return solve_exact(inst)

    if method == "ils":
        iters = DATA.ils_iterations if ils_iterations is None else ils_iterations
        k = DATA.ils_k if ils_k is None else ils_k
        inicial = greedy_add(inst)
        return ILS(k=k, max_iterations=iters, seed=seed).solve(inicial)

    raise ValueError(f"experto desconocido: {method!r}. Opciones: {METHODS}")
