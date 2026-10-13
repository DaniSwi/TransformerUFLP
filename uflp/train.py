"""Ciclo de entrenamiento: entropía cruzada, early stopping e historia.

El early stopping puede vigilar dos cosas distintas:

  select_by="val_loss"     barato, una pasada sobre validación por época.
  select_by="rollout_gap"  mide lo que realmente importa: el costo de las
                           soluciones que el modelo construye. Cuesta unos
                           0,3 s por época sobre 25 instancias.

La distinción no es cosmética. Con la MLP medimos un grid de seis
configuraciones y hubo TRES pares donde la de mayor accuracy dio peor solución:
el accuracy mide coincidencia con el experto en los estados que el experto
visitó, y el rollout transcurre sobre los estados a los que llega el modelo.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TRAIN, TrainConfig
from .rollout import rollout_costs


@dataclass
class History:
    """Lo que hace falta para el gráfico de convergencia de la sección 5.2."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    rollout_gap: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = float("inf")
    stopped_at: int = -1
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "train_loss": self.train_loss, "val_loss": self.val_loss,
            "val_acc": self.val_acc, "rollout_gap": self.rollout_gap,
            "best_epoch": self.best_epoch, "best_score": self.best_score,
            "stopped_at": self.stopped_at, "seconds": self.seconds,
        }


def _evaluate(model, X, Y, criterion, device, batch_size=512):
    """Pérdida y accuracy en validación, por lotes para no reventar memoria."""
    model.eval()
    total_loss, correct, n = 0.0, 0, len(X)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X[i:i + batch_size].to(device)
            yb = Y[i:i + batch_size].to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(xb)
            correct += int((logits.argmax(-1) == yb).sum().item())
    return total_loss / n, correct / n


def train(model, train_data, val_data, cfg: TrainConfig = TRAIN,
          device: str = "cpu", val_instances=None, expert_costs=None,
          verbose: bool = True) -> History:
    """Entrena el modelo y devuelve la historia.

    Parámetros
    ----------
    train_data, val_data : tuplas (X, Y).
    val_instances        : instancias para medir el gap de rollout. Obligatorio
                           si cfg.select_by == "rollout_gap".
    expert_costs         : costos del experto en esas instancias, para el gap.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X_tr, Y_tr = train_data
    X_va, Y_va = val_data
    model = model.to(device)

    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=cfg.batch_size,
                        shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    usa_rollout = cfg.select_by == "rollout_gap"
    if usa_rollout and (val_instances is None or expert_costs is None):
        raise ValueError("select_by='rollout_gap' necesita val_instances y "
                         "expert_costs")

    hist = History()
    mejores_pesos = None
    sin_mejora = 0
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        # ---------------------------------------------------- entrenamiento
        model.train()
        suma = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()                       # 1. borrar gradientes
            logits = model(xb)                          # 2. forward
            loss = criterion(logits, yb)                # 3. pérdida
            loss.backward()                             # 4. backward
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()                            # 5. actualizar pesos
            suma += loss.item() * len(xb)
        train_loss = suma / len(loader.dataset)

        # ------------------------------------------------------- validación
        val_loss, val_acc = _evaluate(model, X_va, Y_va, criterion, device)
        hist.train_loss.append(train_loss)
        hist.val_loss.append(val_loss)
        hist.val_acc.append(val_acc)

        score = val_loss
        gap = float("nan")
        if usa_rollout and epoch % cfg.rollout_every == 0:
            costos = rollout_costs(val_instances, model, device)
            gap = float(100 * np.mean(costos / expert_costs - 1))
            score = gap
        hist.rollout_gap.append(gap)

        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            linea = (f"época {epoch:3d} | train {train_loss:.4f} | "
                     f"val {val_loss:.4f} | acc {val_acc:.3f}")
            if usa_rollout:
                linea += f" | gap {gap:.2f}%"
            print(linea)

        # ---------------------------------------------------- early stopping
        if score < hist.best_score - cfg.min_delta:
            hist.best_score = score
            hist.best_epoch = epoch
            mejores_pesos = copy.deepcopy(model.state_dict())
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= cfg.patience:
                hist.stopped_at = epoch
                if verbose:
                    print(f"early stopping en la época {epoch} "
                          f"(mejor: época {hist.best_epoch}, "
                          f"{cfg.select_by} = {hist.best_score:.4f})")
                break

    if mejores_pesos is not None:
        model.load_state_dict(mejores_pesos)           # restaurar el mejor punto
    hist.seconds = time.perf_counter() - t0
    if hist.stopped_at < 0:
        hist.stopped_at = len(hist.train_loss) - 1
    return hist


def plot_history(hist: History, path: str | None = None):
    """Gráfico de convergencia: pérdida y accuracy por época."""
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tiene_gap = not all(np.isnan(hist.rollout_gap))
    n = 3 if tiene_gap else 2
    fig, ax = plt.subplots(1, n, figsize=(5.4 * n, 4.4))

    ax[0].plot(hist.train_loss, label="entrenamiento")
    ax[0].plot(hist.val_loss, label="validación")
    ax[0].axvline(hist.best_epoch, ls="--", lw=1, color="gray")
    ax[0].set_title("Pérdida (entropía cruzada)")
    ax[0].set_xlabel("época")
    ax[0].legend()

    ax[1].plot(np.array(hist.val_acc) * 100, color="tab:green")
    ax[1].axvline(hist.best_epoch, ls="--", lw=1, color="gray")
    ax[1].set_title("Accuracy de validación (%)")
    ax[1].set_xlabel("época")

    if tiene_gap:
        ax[2].plot(hist.rollout_gap, color="tab:orange")
        ax[2].axvline(hist.best_epoch, ls="--", lw=1, color="gray")
        ax[2].set_title("Gap del rollout vs experto (%)")
        ax[2].set_xlabel("época")

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig
