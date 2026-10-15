"""state2vec: el estado parcial como matriz (M+1, 9).

La fila i corresponde SIEMPRE a la facilidad i. La fila M es un pseudo-elemento
que representa la acción STOP y transporta la información global del estado.

Ninguna columna tiene unidades: los montos se dividen por el costo fijo medio y
las distancias por la distancia media, así que una instancia con costos diez
veces mayores produce exactamente la misma matriz.
"""
from __future__ import annotations

import numpy as np

from .state import UFLP_State

VEC_LEN = 9

# índices de columna, para no escribir números sueltos por ahí
C_STOP = 0        # flag    1 solo en la fila especial
C_COSTO = 1       # propia  f_i / f_medio
C_DIST = 2        # propia  distancia media a los clientes / c_medio
C_AHORRO = 3      # relativa  ahorro de cobertura, 0..1
C_DELTA = 4       # relativa  ΔZ_i / f_medio  (el puntaje de Greedy Add)
C_ULTIMA = 5      # flag    marca la fila que arma la query del decoder
C_MASCARA = 6     # flag    1 si la fila no es una acción legal
C_AVANCE = 7      # global  |S| / M
C_MEJOR = 8       # global  el mejor ΔZ entre las cerradas

COLUMN_NAMES = ("es_stop", "costo_fijo", "dist_media", "ahorro", "delta_z",
                "es_ultima", "no_valida", "avance", "mejor_delta")

# Partición estática / dinámica de las columnas. La usa la variante
# UFLP_AttentionEstatico: las tres primeras no dependen del estado parcial, así
# que su embedding se puede calcular una vez por instancia y reutilizar.
COLS_ESTATICAS = (C_STOP, C_COSTO, C_DIST)
COLS_DINAMICAS = (C_AHORRO, C_DELTA, C_ULTIMA, C_MASCARA, C_AVANCE, C_MEJOR)

ROLES = {
    "es_stop": "flag", "costo_fijo": "propia", "dist_media": "propia",
    "ahorro": "relativa", "delta_z": "relativa", "es_ultima": "flag",
    "no_valida": "flag", "avance": "global", "mejor_delta": "global",
}


def state2vec(state: UFLP_State, ultima: int | None = None) -> np.ndarray:
    """Codifica un estado parcial como np.ndarray (M+1, VEC_LEN) float32.

    Parámetros
    ----------
    state  : el estado parcial.
    ultima : índice de la última instalación abierta. None si S = {}, en cuyo
             caso la fila especial toma el flag `es_ultima` para que siempre
             haya exactamente una fila marcada.
    """
    inst = state.inst_info
    c, f = inst.cost_matrix, inst.fixed_costs
    M = inst.num_facilities

    f_bar = f.mean() + 1e-12
    c_bar = c.mean() + 1e-12

    # Referencia de servicio actual. Con S = {} el costo es infinito, así que se
    # usa el peor servicio posible: el ahorro se mide contra ese peor caso.
    ref = np.where(np.isfinite(state.best_cost), state.best_cost, c.max(axis=0))

    abierta = np.zeros(M)
    if state.opened:
        abierta[list(state.opened)] = 1.0

    ahorro = np.maximum(0.0, ref[None, :] - c).sum(axis=1)
    ahorro[abierta == 1] = 0.0
    delta = np.clip((f - ahorro) / f_bar, -10, 10)

    cerradas = abierta == 0
    mejor = float(delta[cerradas].min()) if cerradas.any() else 0.0

    v = np.zeros((M + 1, VEC_LEN), dtype=np.float32)

    v[:M, C_COSTO] = f / f_bar
    v[:M, C_DIST] = c.mean(axis=1) / c_bar
    v[:M, C_AHORRO] = ahorro / (ahorro.max() + 1e-12)
    v[:M, C_DELTA] = delta
    v[:M, C_MASCARA] = abierta                     # ya abierta -> acción inválida

    v[M, C_STOP] = 1.0                             # la fila especial
    v[M, C_MASCARA] = 1.0 if not state.opened else 0.0   # no parar en S = {}

    if ultima is not None:
        v[ultima, C_ULTIMA] = 1.0
    else:
        v[M, C_ULTIMA] = 1.0

    v[:, C_AVANCE] = len(state.opened) / M
    v[:, C_MEJOR] = mejor
    return v


def describe(v: np.ndarray, elegida: int | None = None) -> str:
    """Imprime la matriz legible. Útil para el sanity check del notebook."""
    M = len(v) - 1
    out = ["fila  " + "".join(f"{n[:10]:>12}" for n in COLUMN_NAMES),
           "-" * (6 + 12 * VEC_LEN)]
    for i, fila in enumerate(v):
        nombre = "STOP" if i == M else str(i)
        marca = "  <- elegida" if elegida is not None and i == elegida else ""
        out.append(f"{nombre:>4}  " + "".join(f"{x:>12.2f}" for x in fila) + marca)
    return "\n".join(out)
