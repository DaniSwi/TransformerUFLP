"""Evaluación: gaps contra el óptimo, media y desviación, y por tamaño."""
from __future__ import annotations

import random
import time

import numpy as np

from .agents import ILS, greedy_add, hill_climbing, random_constructive
from .exact import solve_exact
from .instance import random_instances
from .rollout import solve_multistart, solve_with_model

#: Métodos disponibles. Cada uno: instancia -> UFLP_State.
#: El modelo se inyecta con `make_methods`, porque necesita los pesos.
BASE_METHODS = ("Aleatorio", "Greedy Add", "Greedy + HC", "ILS", "Óptimo (MIP)")


def make_methods(model=None, device: str = "cpu", multistart: int = 0,
                 ils_iterations: int = 25, with_exact: bool = True) -> dict:
    """Construye el diccionario de métodos a comparar."""
    metodos = {
        "Aleatorio": lambda inst: random_constructive(inst, random.Random(0)),
        "Greedy Add": greedy_add,
        "Greedy + HC": lambda inst: hill_climbing(greedy_add(inst)),
    }
    if model is not None:
        metodos["Modelo"] = lambda inst: solve_with_model(inst, model, device)
        metodos["Modelo + HC"] = lambda inst: hill_climbing(
            solve_with_model(inst, model, device))
        if multistart:
            metodos[f"Modelo x{multistart}"] = lambda inst: solve_multistart(
                inst, model, multistart, device)
    metodos["ILS"] = lambda inst: ILS(
        max_iterations=ils_iterations, seed=1).solve(greedy_add(inst))
    if with_exact:
        metodos["Óptimo (MIP)"] = solve_exact
    return metodos


def benchmark(instances, methods: dict, reference: str = "Óptimo (MIP)"):
    """Corre todos los métodos y devuelve costos, tiempos y gaps.

    Devuelve
    --------
    dict con 'costs' {método: array}, 'times' {método: segundos/instancia} y
    'gaps' {método: array de gaps porcentuales contra `reference`}.
    """
    costs = {k: [] for k in methods}
    times = {k: 0.0 for k in methods}

    for inst in instances:
        for nombre, fn in methods.items():
            t0 = time.perf_counter()
            estado = fn(inst)
            times[nombre] += time.perf_counter() - t0
            costs[nombre].append(estado.cost)

    costs = {k: np.array(v) for k, v in costs.items()}
    times = {k: v / len(instances) for k, v in times.items()}

    if reference not in costs:
        reference = "ILS" if "ILS" in costs else next(iter(costs))
    ref = costs[reference]
    gaps = {k: 100 * (v / ref - 1) for k, v in costs.items()}

    return {"costs": costs, "times": times, "gaps": gaps,
            "reference": reference, "n": len(instances)}


def format_table(res: dict, title: str = "") -> str:
    """Tabla con media y desviación estándar del gap."""
    lineas = []
    if title:
        lineas.append(title)
    lineas.append(f"{'Método':<16}{'Z medio':>11}{'gap medio':>12}"
                  f"{'desv. est.':>12}{'t medio':>11}")
    lineas.append("-" * 62)
    for k in res["costs"]:
        g = res["gaps"][k]
        lineas.append(f"{k:<16}{res['costs'][k].mean():>11.2f}"
                      f"{g.mean():>11.2f}%{g.std():>11.2f}%"
                      f"{res['times'][k]:>10.3f}s")
    lineas.append(f"(referencia: {res['reference']} · {res['n']} instancias "
                  f"nuevas)")
    return "\n".join(lineas)


def evaluate_by_size(sizes, methods_fn, n_instances: int = 20,
                     n_clients: int = 200, seed: int = 999,
                     reference: str = "Óptimo (MIP)", verbose: bool = True):
    """Corre el benchmark en varios tamaños de instancia.

    `methods_fn` recibe el tamaño y devuelve el diccionario de métodos; así el
    mismo modelo (sin reentrenar) se evalúa en M = 5, 10, 20, 50...
    """
    resultados = {}
    for k, M in enumerate(sizes):
        instancias = random_instances(n_instances, M, n_clients,
                                      seed=seed + 1000 * k)
        res = benchmark(instancias, methods_fn(M), reference=reference)
        resultados[M] = res
        if verbose:
            print(format_table(res, f"\n--- M = {M}, N = {n_clients} ---"))
    return resultados


def plot_by_size(resultados: dict, methods=None, path: str | None = None):
    """Gap medio +- desviación por tamaño de instancia (sección 5.4)."""
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted(resultados)
    if methods is None:
        methods = [m for m in resultados[sizes[0]]["gaps"]
                   if m != resultados[sizes[0]]["reference"]]

    x = np.arange(len(sizes))
    ancho = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(1.9 * len(sizes) + 4, 4.8))

    for i, m in enumerate(methods):
        medias = [resultados[s]["gaps"][m].mean() for s in sizes]
        desv = [resultados[s]["gaps"][m].std() for s in sizes]
        ax.bar(x + i * ancho, medias, ancho, yerr=desv, capsize=3, label=m)

    ax.set_xticks(x + ancho * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"M = {s}" for s in sizes])
    ax.set_ylabel("gap vs óptimo (%)")
    ax.set_title("Desempeño por tamaño de instancia")
    ax.axhline(0, color="gray", lw=1)
    ax.legend()
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig
