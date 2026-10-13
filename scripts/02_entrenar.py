#!/usr/bin/env python3
"""Entrena el encoder-decoder y guarda los pesos y la curva de convergencia.

    python scripts/02_entrenar.py
    python scripts/02_entrenar.py --select-by rollout_gap --epochs 60
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uflp import (DATA, MODEL, TRAIN, UFLP_Attention, load_datasets,  # noqa: E402
                  plot_history, random_instances, solve_expert, train)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-d", "--data", default="data/imitacion.pt")
    p.add_argument("-o", "--out", default="runs/modelo.pt")
    p.add_argument("--epochs", type=int, default=TRAIN.epochs)
    p.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    p.add_argument("--lr", type=float, default=TRAIN.lr)
    p.add_argument("--dropout", type=float, default=MODEL.dropout)
    p.add_argument("--layers", type=int, default=MODEL.n_encoder_layers)
    p.add_argument("--select-by", default=TRAIN.select_by,
                   choices=["val_loss", "rollout_gap"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()

    mcfg = replace(MODEL, dropout=args.dropout, n_encoder_layers=args.layers)
    tcfg = replace(TRAIN, epochs=args.epochs, batch_size=args.batch_size,
                   lr=args.lr, select_by=args.select_by)

    datos = load_datasets(args.data)
    print(f"train {tuple(datos['train'][0].shape)} · "
          f"val {tuple(datos['val'][0].shape)}")

    model = UFLP_Attention(mcfg)
    print(f"\nparámetros entrenables: {model.n_parameters:,}")
    for k, v in model.parameter_breakdown().items():
        print(f"  {k:<16}{v:>10,}  {100 * v / model.n_parameters:5.1f}%")

    # instancias para medir el gap de rollout durante el entrenamiento
    val_inst, costos_exp = None, None
    if tcfg.select_by == "rollout_gap":
        val_inst = random_instances(tcfg.rollout_instances, DATA.n_facilities,
                                    DATA.n_clients, seed=DATA.seed_val)
        costos_exp = np.array([solve_expert(i, DATA.expert, seed=k).cost
                               for k, i in enumerate(val_inst)])

    print()
    hist = train(model, datos["train"], datos["val"], tcfg, args.device,
                 val_instances=val_inst, expert_costs=costos_exp)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "model_config": asdict(mcfg),
                "train_config": asdict(tcfg),
                "history": hist.as_dict()}, out)
    (out.parent / "history.json").write_text(
        json.dumps(hist.as_dict(), indent=1), encoding="utf-8")
    plot_history(hist, str(out.parent / "convergencia.png"))

    print(f"\nentrenado en {hist.seconds:.0f} s · mejor época {hist.best_epoch}")
    print(f"pesos    -> {out}")
    print(f"curva    -> {out.parent / 'convergencia.png'}")


if __name__ == "__main__":
    main()
