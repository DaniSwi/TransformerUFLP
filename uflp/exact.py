"""Óptimo exacto del UFLP con un solver MIP (HiGHS, vía scipy).

La relajación LP de la formulación desagregada es muy ajustada, así que estas
instancias se resuelven a optimalidad en décimas de segundo:

    M =  20  ->  0.10 s        M =  50  ->  0.19 s
    M =  30  ->  0.10 s        M = 100  ->  0.41 s

Las x_ij se dejan continuas: dado y binario, la asignación óptima ya es entera.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix

from .instance import UFLP_Instance
from .state import UFLP_State


def solve_exact(inst: UFLP_Instance) -> UFLP_State:
    """Devuelve el estado óptimo. Lanza RuntimeError si el solver falla.

        min  sum_i f_i y_i + sum_ij c_ij x_ij
        s.a. sum_i x_ij = 1        para todo cliente j
             x_ij - y_i <= 0       para todo (i, j)
             y binaria, x >= 0
    """
    M, N = inst.num_facilities, inst.num_clients
    c, f = inst.cost_matrix, inst.fixed_costs
    nx = M * N
    costo = np.concatenate([f, c.ravel()])          # [y (M) , x (M*N)]

    # cada cliente asignado exactamente una vez
    A1 = lil_matrix((N, M + nx))
    for j in range(N):
        A1[j, M + np.arange(M) * N + j] = 1
    con_asign = LinearConstraint(csr_matrix(A1), 1, 1)

    # no se puede servir desde una instalación cerrada
    A2 = lil_matrix((nx, M + nx))
    filas = np.arange(nx)
    A2[filas, M + filas] = 1
    A2[filas, np.repeat(np.arange(M), N)] = -1
    con_abierta = LinearConstraint(csr_matrix(A2), -np.inf, 0)

    res = milp(
        c=costo,
        constraints=[con_asign, con_abierta],
        integrality=np.concatenate([np.ones(M), np.zeros(nx)]),
        bounds=Bounds(0, 1),
    )
    if not res.success:
        raise RuntimeError(f"el solver MIP no encontró solución: {res.message}")

    S = np.where(res.x[:M] > 0.5)[0].tolist()
    return UFLP_State(inst, opened=S)
