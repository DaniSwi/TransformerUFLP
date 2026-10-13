#!/usr/bin/env python3
"""Genera el dataset de imitación y lo guarda en disco.

    python scripts/01_generar_datos.py
    python scripts/01_generar_datos.py --expert exact --n-train 500
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uflp import DATA, build_datasets, save_datasets  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expert", default=DATA.expert,
                   choices=["ils", "exact", "greedy"])
    p.add_argument("-M", "--n-facilities", type=int, default=DATA.n_facilities)
    p.add_argument("-N", "--n-clients", type=int, default=DATA.n_clients)
    p.add_argument("--n-train", type=int, default=DATA.n_train)
    p.add_argument("--n-val", type=int, default=DATA.n_val)
    p.add_argument("--n-test", type=int, default=DATA.n_test)
    p.add_argument("-o", "--out", default="data/imitacion.pt")
    args = p.parse_args()

    cfg = replace(DATA, expert=args.expert, n_facilities=args.n_facilities,
                  n_clients=args.n_clients, n_train=args.n_train,
                  n_val=args.n_val, n_test=args.n_test)

    print(f"M = {cfg.n_facilities} · N = {cfg.n_clients} · "
          f"experto = {cfg.expert}\n")
    t0 = time.perf_counter()
    partes = build_datasets(cfg)
    ruta = save_datasets(partes, args.out)
    print(f"\ngenerado en {time.perf_counter() - t0:.0f} s -> {ruta}")


if __name__ == "__main__":
    main()
