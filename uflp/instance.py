"""Instancias del UFLP: los datos invariantes del problema."""
from __future__ import annotations

import numpy as np

from .config import DATA


class UFLP_Instance:
    """Invariantes del problema. Solo lectura durante toda la búsqueda.

    Atributos
    ---------
    facility_locations : (M, 2)   coordenadas de las ubicaciones candidatas
    client_locations   : (N, 2)   coordenadas de los clientes
    fixed_costs        : (M,)     f_i, costo de abrir la instalación i
    cost_matrix        : (M, N)   c_ij, costo de servir al cliente j desde i
    """

    __slots__ = ("facility_locations", "client_locations", "fixed_costs",
                 "num_facilities", "num_clients", "cost_matrix")

    def __init__(self, facility_locations, client_locations, fixed_costs):
        self.facility_locations = np.asarray(facility_locations, dtype=np.float64)
        self.client_locations = np.asarray(client_locations, dtype=np.float64)
        self.fixed_costs = np.asarray(fixed_costs, dtype=np.float64)
        self.num_facilities = len(self.facility_locations)
        self.num_clients = len(self.client_locations)

        d = self.facility_locations[:, None, :] - self.client_locations[None, :, :]
        self.cost_matrix = np.sqrt((d ** 2).sum(-1))

    def __repr__(self):
        return (f"UFLP_Instance(M={self.num_facilities}, "
                f"N={self.num_clients})")


def random_instance(M: int | None = None, N: int | None = None,
                    rng: np.random.Generator | None = None,
                    f_lo: float | None = None,
                    f_hi: float | None = None) -> UFLP_Instance:
    """Instancia euclidiana 2D en el cuadrado unitario.

    Los costos fijos se escalan con sqrt(N) para que abrir una instalación sea
    una decisión no trivial: ni tan barata que convenga abrirlas todas, ni tan
    cara que convenga abrir una sola.
    """
    M = DATA.n_facilities if M is None else M
    N = DATA.n_clients if N is None else N
    f_lo = DATA.fixed_cost_lo if f_lo is None else f_lo
    f_hi = DATA.fixed_cost_hi if f_hi is None else f_hi
    rng = np.random.default_rng() if rng is None else rng

    facilities = rng.random((M, 2))
    clients = rng.random((N, 2))
    scale = np.sqrt(N) * 0.10
    fixed = rng.uniform(f_lo, f_hi, size=M) * scale
    return UFLP_Instance(facilities, clients, fixed)


def random_instances(n: int, M: int | None = None, N: int | None = None,
                     seed: int = 0, **kwargs) -> list[UFLP_Instance]:
    """n instancias de un mismo tamaño, reproducibles a partir de `seed`.

    Dos llamadas con semillas distintas generan conjuntos disjuntos: es así como
    se separan entrenamiento, validación y prueba.
    """
    rng = np.random.default_rng(seed)
    return [random_instance(M, N, rng, **kwargs) for _ in range(n)]
