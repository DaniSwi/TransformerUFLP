"""Generación de los datos de imitación.

El experto devuelve un CONJUNTO S*, y la imitación paso a paso necesita una
SECUENCIA. Le imponemos el orden voraz restringido a S*: en cada paso se abre,
entre las instalaciones de S* que faltan, la de mayor ahorro inmediato.

Es determinista, así que estados parecidos nunca reciben etiquetas
contradictorias. Y es el orden que seguiría naturalmente un agente constructivo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import DATA
from .environment import UFLP_Environment as env
from .experts import solve_expert
from .features import VEC_LEN, state2vec
from .instance import UFLP_Instance, random_instance
from .state import UFLP_State


def decompose(inst: UFLP_Instance, S_star) -> list[tuple[np.ndarray, int]]:
    """Descompone S* en pares (state2vec(estado), índice de la fila elegida).

    Un episodio con |S*| = k produce k + 1 muestras: las k aperturas más la
    decisión de detenerse, cuyo target es la clase M (la fila STOP).
    """
    M = inst.num_facilities
    state = UFLP_State(inst)
    ultima = None
    faltan = set(S_star)
    pares = []

    while faltan:
        # acción escogida: la de S* con mayor ahorro inmediato
        elegida = min(faltan, key=lambda i:
                      env.calculate_cost_after_action(state, ("constructive", i)))
        pares.append((state2vec(state, ultima), elegida))
        state = env.state_transition(state, ("constructive", elegida))
        ultima = elegida
        faltan.discard(elegida)

    pares.append((state2vec(state, ultima), M))        # última decisión: STOP
    return pares


def generate_imitation_data(n_instances: int, M: int | None = None,
                            N: int | None = None, seed: int = 0,
                            expert: str | None = None,
                            verbose: bool = False):
    """Devuelve (X, Y) con X float32 (n_muestras, M+1, 9) e Y long.

    Dos llamadas con semillas distintas generan instancias disjuntas: así se
    separan entrenamiento, validación y prueba. Nunca se divide por muestra,
    porque las muestras de una misma instancia comparten toda su geometría.
    """
    rng = np.random.default_rng(seed)
    X: list[np.ndarray] = []
    Y: list[int] = []

    for t in range(n_instances):
        inst = random_instance(M, N, rng)
        S_star = solve_expert(inst, expert, seed=seed * 100_000 + t).opened
        for x, y in decompose(inst, S_star):
            X.append(x)
            Y.append(y)
        if verbose and (t + 1) % 100 == 0:
            print(f"  {t + 1}/{n_instances} instancias · {len(X)} muestras")

    return (torch.from_numpy(np.stack(X)),
            torch.tensor(np.array(Y), dtype=torch.long))


def build_datasets(cfg=DATA, verbose: bool = True):
    """Genera los tres conjuntos según la configuración."""
    partes = {}
    for nombre, n, seed in (("train", cfg.n_train, cfg.seed_train),
                            ("val", cfg.n_val, cfg.seed_val),
                            ("test", cfg.n_test, cfg.seed_test)):
        if verbose:
            print(f"[{nombre}] {n} instancias, experto = {cfg.expert}")
        partes[nombre] = generate_imitation_data(
            n, cfg.n_facilities, cfg.n_clients, seed=seed,
            expert=cfg.expert, verbose=verbose)
        if verbose:
            x, y = partes[nombre]
            print(f"  -> X {tuple(x.shape)}  Y {tuple(y.shape)}  "
                  f"STOP {100 * (y == cfg.n_facilities).float().mean():.1f}%")
    return partes


def save_datasets(partes: dict, path: str | Path):
    """Guarda los tres conjuntos en un único archivo .pt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: {"X": x, "Y": y} for k, (x, y) in partes.items()}, path)
    return path


def load_datasets(path: str | Path) -> dict:
    bruto = torch.load(Path(path), weights_only=True)
    return {k: (v["X"], v["Y"]) for k, v in bruto.items()}
