"""
train_yolo.py - Etapas 3 y 4: entrena YOLO26 con Ultralytics y guarda un resumen comparable.

  --mode scratch   arquitectura desde el .yaml, SIN pesos preentrenados (Etapa 3)
  --mode transfer  parte de pesos preentrenados en COCO (.pt) (Etapa 4)

Uso (desde la carpeta del proyecto):
    python src/train_yolo.py --mode scratch --epochs 3 --name smoke       # prueba rápida
    python src/train_yolo.py --mode scratch --epochs 150 --name yolo26s_scratch

Usa solo train y val. El test no se toca aquí. Resultados en runs/yolo/<name>/
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scratch", "transfer"], required=True)
    ap.add_argument("--model", default="yolo26s")
    ap.add_argument("--name", default=None)
    ap.add_argument("--yaml", default="data/fjord/data.yaml")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--freeze", type=int, default=0, help="capas iniciales congeladas (solo transfer)")
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plots", type=int, default=1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    name = args.name or f"{args.model}_{args.mode}"
    project = Path("runs/yolo").resolve()      # ruta ABSOLUTA: evita que Ultralytics anide carpetas
    weights = f"{args.model}.yaml" if args.mode == "scratch" else f"{args.model}.pt"
    model = YOLO(weights)
    kw = dict(data=str(Path(args.yaml).resolve()), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
              workers=args.workers, project=str(project), name=name, exist_ok=True, seed=args.seed,
              patience=args.patience, plots=bool(args.plots), device=args.device)
    if args.mode == "scratch":
        kw["pretrained"] = False
    else:
        kw["freeze"] = args.freeze or None
    t0 = time.time()
    model.train(**kw)
    total_min = (time.time() - t0) / 60

    save_dir = Path(model.trainer.save_dir)
    h = pd.read_csv(save_dir / "results.csv")
    h.columns = [c.strip() for c in h.columns]
    fit = 0.1 * h["metrics/mAP50(B)"] + 0.9 * h["metrics/mAP50-95(B)"]
    best = int(fit.values.argmax())
    params = sum(p.numel() for p in model.trainer.model.parameters())
    summary = dict(name=name, mode=args.mode, model=args.model, params=params,
                   epochs_requested=args.epochs, epochs_run=len(h), best_epoch=int(h["epoch"].iloc[best]),
                   best_fitness=float(fit.iloc[best]), val_map50_at_best=float(h["metrics/mAP50(B)"].iloc[best]),
                   val_map50_95_at_best=float(h["metrics/mAP50-95(B)"].iloc[best]), total_time_min=total_min,
                   train_time_min=float(h["time"].iloc[-1]) / 60 if "time" in h else None,
                   args={k: v for k, v in vars(args).items()}, save_dir=str(save_dir))
    json.dump(summary, open(save_dir / "summary.json", "w"), indent=2)
    print("\n=== RESUMEN ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
