"""Aprendizaje por imitación para el Uncapacitated Facility Location Problem.

    from uflp import random_instance, greedy_add, solve_expert
    from uflp import UFLP_Attention, train, solve_with_model

Módulos
-------
config       hiperparámetros, en un solo lugar
instance     UFLP_Instance, generación de instancias
state        UFLP_State, con los cachés best_cost / second_cost
environment  acciones, deltas en O(N), transiciones
agents       Greedy Add, Hill Climbing, ILS, baseline aleatorio
exact        óptimo con un solver MIP
experts      el experto a imitar, elegible por string
features     state2vec: el estado como matriz (M+1, 9)
data         descomposición de soluciones y dataset de imitación
model        encoder-decoder con puntero
train        ciclo de entrenamiento con early stopping
rollout      el modelo dentro del agente constructivo
evaluate     benchmark, gaps y desempeño por tamaño
"""
from .agents import (ILS, GreedyAgent, LocalSearchAgent, SingleAgentSolver,
                     evalConstructiveActions, greedy_add, hill_climbing,
                     random_constructive)
from .config import DATA, MODEL, TRAIN, DataConfig, ModelConfig, TrainConfig
from .data import (build_datasets, decompose, generate_imitation_data,
                   load_datasets, save_datasets)
from .environment import UFLP_Environment, env
from .evaluate import (benchmark, evaluate_by_size, format_table, make_methods,
                       plot_by_size)
from .exact import solve_exact
from .experts import solve_expert
from .features import COLUMN_NAMES, VEC_LEN, describe, state2vec
from .instance import UFLP_Instance, random_instance, random_instances
from .model import UFLP_Attention
from .rollout import rollout_costs, solve_multistart, solve_with_model
from .state import UFLP_State
from .train import History, plot_history, train

__version__ = "1.0.0"

__all__ = [
    "DATA", "MODEL", "TRAIN", "DataConfig", "ModelConfig", "TrainConfig",
    "UFLP_Instance", "random_instance", "random_instances",
    "UFLP_State", "UFLP_Environment", "env",
    "GreedyAgent", "LocalSearchAgent", "SingleAgentSolver", "ILS",
    "evalConstructiveActions", "greedy_add", "hill_climbing",
    "random_constructive", "solve_exact", "solve_expert",
    "VEC_LEN", "COLUMN_NAMES", "state2vec", "describe",
    "decompose", "generate_imitation_data", "build_datasets",
    "save_datasets", "load_datasets",
    "UFLP_Attention", "train", "History", "plot_history",
    "solve_with_model", "solve_multistart", "rollout_costs",
    "make_methods", "benchmark", "format_table", "evaluate_by_size",
    "plot_by_size",
]
