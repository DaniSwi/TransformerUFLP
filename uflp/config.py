"""Toda la configuración del proyecto en un solo lugar.

Las tres dataclasses de abajo son la fuente de verdad de los hiperparámetros:
la tabla de la sección 5.1 de la presentación sale directamente de aquí.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Generación de instancias y del dataset de imitación."""

    # --- instancias ---
    n_facilities: int = 20          # M
    n_clients: int = 200            # N
    fixed_cost_lo: float = 0.2      # rango del multiplicador del costo fijo
    fixed_cost_hi: float = 3.0

    # --- split, SEPARADO POR INSTANCIA (nunca por muestra) ---
    n_train: int = 1500
    n_val: int = 200
    n_test: int = 200
    seed_train: int = 0
    seed_val: int = 7
    seed_test: int = 13

    # --- experto a imitar ---
    expert: str = "ils"             # "ils" | "exact" | "greedy"
    ils_iterations: int = 25
    ils_k: int = 2                  # swaps por perturbación


@dataclass(frozen=True)
class ModelConfig:
    """Arquitectura encoder-decoder con puntero (Kool et al., adaptada)."""

    vec_len: int = 9                # features por fila (lo fija features.py)
    d_model: int = 128              # d
    n_heads: int = 8                # cabezas del MHA (encoder y glimpse)
    n_encoder_layers: int = 3       # capas del encoder
    d_ff: int = 512                 # 4 x d
    dropout: float = 0.1
    clip_c: float = 10.0            # C del tanh en el puntero
    d_dinamico: int = 64            # sólo UFLP_AttentionEstatico: ancho
                                    # del embedding de lo que cambia


@dataclass(frozen=True)
class TrainConfig:
    """Ciclo de entrenamiento."""

    epochs: int = 200               # tope; el corte real lo pone el early stopping
    batch_size: int = 128
    lr: float = 1e-4
    grad_clip: float = 1.0          # norma máxima del gradiente
    patience: int = 15
    min_delta: float = 1e-3

    # "val_loss" es barato; "rollout_gap" selecciona por la métrica que importa
    select_by: str = "val_loss"
    rollout_instances: int = 25     # instancias para medir el gap cada época
    rollout_every: int = 1

    seed: int = 0


DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
