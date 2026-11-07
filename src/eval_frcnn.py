"""
eval_frcnn.py - Evalúa Faster R-CNN con NUESTRO evaluador común (el mismo que los otros tres modelos).

    python src/eval_frcnn.py --name frcnn_v1 --split val
    python src/eval_frcnn.py --name frcnn_v1 --split test     # resultado final (usar UNA vez)

El umbral de confianza para precisión/recall se elige SIEMPRE en val (máx. F1) y se aplica tal cual al test.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from detkit.metrics import best_f1_threshold
from detkit.report import draw_predictions, full_report, save_predictions
from frcnn.lib import FjordFRCNN, build_model, collate, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="frcnn_v1")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--yaml", default="data/fjord/data.yaml")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out = Path(args.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res_dir, fig_dir, pred_dir = out / "results" / "eval", out / "figures", out / "results" / "preds"
    for d in (res_dir, fig_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    run = out / "runs" / "frcnn" / args.name
    ck = torch.load(run / "best.pt", map_location=dev)
    # Se construye igual que en el entrenamiento (las BatchNorm congeladas deben coincidir) y luego se cargan los pesos
    model = build_model(ck["num_classes"], ck["pretrained"], 3, ck["img_size"], ck["anchors"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    names = ck["names"]
    print(f"Modelo {args.name}: mejor época {ck['epoch']}")

    def run_split(split):
        ds = FjordFRCNN(args.yaml, split, train=False)
        dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=args.workers)
        return (ds, *predict(model, dl, dev))

    ds_v, pv, gv, sv = run_split("val")
    best = best_f1_threshold(pv, gv, len(names))
    thr = best["thr"]
    print(f"Umbral de confianza elegido en VAL (máx. F1): {thr:.2f}  (F1 val = {best['f1']:.3f})")
    json.dump(best | {"tp": None, "fp": None, "fn": None}, open(res_dir / f"{args.name}_val_threshold.json", "w"), indent=2)

    # velocidad provisional: una imagen por vez, desde un tensor ya cargado hasta las cajas finales
    xs = [ds_v[i][0].to(dev).float() / 255 for i in range(min(60, len(ds_v)))]
    with torch.no_grad():
        for x in xs[:10]:
            model([x])
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for x in xs[10:]:
            model([x])
        if dev == "cuda":
            torch.cuda.synchronize()
    ms = (time.time() - t0) / max(len(xs) - 10, 1) * 1000
    summ = json.load(open(run / "summary.json"))
    extra = dict(model=args.name, params_M=summ["params"] / 1e6, train_time_min=summ["total_time_min"],
                 best_epoch=summ["best_epoch"], inference_ms_per_image=ms, fps=1000 / ms)

    splits = {"val": (ds_v, pv, gv, sv)}
    if args.split == "test":
        splits["test"] = run_split("test")
    for sp, (ds, p, g, st) in splits.items():
        s, tab = full_report(f"{args.name}_{sp}", p, g, names, thr, res_dir, extra)
        save_predictions(pred_dir / f"{args.name}_{sp}.csv", st, p, names)
        draw_predictions(ds.files, st, p, g, names, max(thr, 0.3), fig_dir / f"{args.name}_{sp}_preds.jpg")
        print(f"\n=== {args.name} en {sp.upper()} ({s['n_images']} imágenes, {s['n_objects']} objetos, "
              f"{s['n_classes_evaluated']} clases evaluadas) ===")
        print(f"mAP@0.5 = {s['map50']:.3f} | mAP@0.5:0.95 = {s['map50_95']:.3f} | mAP@0.75 = {s['map75']:.3f}")
        print(f"Precisión = {s['precision']:.3f} | Recall = {s['recall']:.3f} | F1 = {s['f1']:.3f}  (umbral {thr:.2f})")
        print(tab.round(3).to_string(index=False))
    print(f"\nInferencia (provisional): {ms:.1f} ms por imagen ({1000 / ms:.0f} FPS) en {dev}")
    print("Archivos en results/eval/, results/preds/ y figures/")


if __name__ == "__main__":
    main()
