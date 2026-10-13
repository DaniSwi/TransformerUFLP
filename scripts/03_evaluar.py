#!/usr/bin/env python3
"""Evalúa el modelo entrenado: tabla comparativa y desempeño por tamaño.

    python scripts/03_evaluar.py
    python scripts/03_evaluar.py --sizes 5 10 20 50 --n-instances 20
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uflp import (DATA, MODEL, UFLP_Attention, benchmark,  # noqa: E402
                  evaluate_by_size, format_table, make_methods, plot_by_size,
                  random_instances)


def cargar_modelo(ruta: str, device: str):
    ckpt = torch.load(ruta, map_location=device, weights_only=False)
    cfg = replace(MODEL, **ckpt.get("model_config", {}))
    model = UFLP_Attention(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-m", "--model", default="runs/modelo.pt")
    p.add_argument("-o", "--out-dir", default="runs")
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[5, 10, 15, 20, 30, 50])
    p.add_argument("--n-instances", type=int, default=20)
    p.add_argument("--multistart", type=int, default=0)
    p.add_argument("--no-exact", action="store_true",
                   help="usa el ILS como referencia en vez del óptimo")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model = cargar_modelo(args.model, args.device)
    referencia = "ILS" if args.no_exact else "Óptimo (MIP)"

    def metodos(_M=None):
        return make_methods(model, args.device, args.multistart,
                            with_exact=not args.no_exact)

    # ---- tabla principal, en el tamaño de entrenamiento -------------------
    inst = random_instances(args.n_instances, DATA.n_facilities,
                            DATA.n_clients, seed=999)
    res = benchmark(inst, metodos(), reference=referencia)
    print(format_table(res, f"\n=== M = {DATA.n_facilities}, "
                            f"N = {DATA.n_clients} (entrenamiento) ==="))

    # ---- desempeño por tamaño --------------------------------------------
    print("\n\n=== Desempeño por tamaño de instancia (zero-shot) ===")
    por_tamano = evaluate_by_size(args.sizes, metodos, args.n_instances,
                                  DATA.n_clients, reference=referencia)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ruta = plot_by_size(por_tamano, path=str(out / "por_tamano.png"))
    print(f"\ngráfico -> {ruta}")


if __name__ == "__main__":
    main()
