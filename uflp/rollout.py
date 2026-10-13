"""El modelo dentro del agente constructivo.

Entrenamiento y rollout no funcionan igual. En entrenamiento cada par
(estado, acción) es una muestra independiente y nada se realimenta: es teacher
forcing. En el rollout la construcción es secuencial, y el lazo pasa por el
ambiente, no por dentro de la red: lo que se realimenta es el índice elegido,
no la distribución.

La red no arrastra estado oculto entre pasos. Toda la memoria de lo construido
vive en las features.
"""
from __future__ import annotations

import numpy as np
import torch

from .environment import UFLP_Environment as env
from .features import state2vec
from .instance import UFLP_Instance
from .state import UFLP_State


@torch.no_grad()
def solve_with_model(inst: UFLP_Instance, model, device: str = "cpu",
                     sample: bool = False,
                     generator: torch.Generator | None = None) -> UFLP_State:
    """Construye una solución completa guiada por el modelo.

    sample=False  toma el argmax (determinista).
    sample=True   muestrea de la softmax, para multistart.
    """
    model.eval()
    M = inst.num_facilities
    state = UFLP_State(inst)
    ultima = None

    for _ in range(M):                      # cota dura: nunca más de M aperturas
        x = torch.from_numpy(state2vec(state, ultima)).unsqueeze(0).to(device)
        logits = model(x)[0]

        if sample:
            probs = torch.softmax(logits, dim=-1)
            idx = int(torch.multinomial(probs, 1, generator=generator).item())
        else:
            idx = int(torch.argmax(logits).item())

        if idx == M:                        # apuntó a la fila STOP
            break

        state = env.state_transition(state, ("constructive", idx))
        ultima = idx

    return state


@torch.no_grad()
def solve_multistart(inst: UFLP_Instance, model, n_samples: int = 32,
                     device: str = "cpu", seed: int = 0) -> UFLP_State:
    """Muestrea varias construcciones y se queda con la mejor.

    Casi gratis y suele ganar algunas décimas sobre el argmax.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    mejor = solve_with_model(inst, model, device)          # el argmax compite
    for _ in range(n_samples):
        cand = solve_with_model(inst, model, device, sample=True, generator=g)
        if cand.cost < mejor.cost:
            mejor = cand
    return mejor


@torch.no_grad()
def rollout_costs(instances, model, device: str = "cpu") -> np.ndarray:
    """Costo de la solución del modelo en cada instancia."""
    return np.array([solve_with_model(i, model, device).cost for i in instances])
