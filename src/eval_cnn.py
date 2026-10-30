"""
eval_cnn.py - Paso 2.3: evalúa la CNN propia en val o test y genera tablas, gráficos y predicciones.

Uso (desde la carpeta del proyecto):
    python src/eval_cnn.py --name cnn_v1 --split val     # análisis y depuración (permitido siempre)
    python src/eval_cnn.py --name cnn_v1 --split test    # resultado final (usar UNA vez)

El umbral de confianza para precisión/recall se elige SIEMPRE en val (máximo F1) y se aplica tal cual al test.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnndet.data import FjordDataset, collate
from cnndet.losses import decode
from cnndet.model import CenterDet
from cnndet.predict import predict_loader
from detkit.metrics import best_f1_threshold
from detkit.report import draw_predictions, full_report, save_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cnn_v1")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--yaml", default="data/fjord/data.yaml")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--score-thr", type=float, default=0.001)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out = Path(args.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res_dir, fig_dir, pred_dir = out / "results" / "eval", out / "figures", out / "results" / "preds"
    for d in (res_dir, fig_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    ck = torch.load(out / "runs" / "cnn" / args.name / "best.pt", map_location=dev)
    model = CenterDet(ck["nc"], width=ck["width"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    names, size = ck["names"], ck["img_size"]
    print(f"Modelo {args.name}: mejor época {ck['epoch']} | {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parámetros")

    def run(split):
        ds = FjordDataset(args.yaml, split, train=False, img_size=size)
        dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate, num_workers=args.workers)
        preds, gts, stems = predict_loader(model, dl, dev, score_thr=args.score_thr)
        return ds, preds, gts, stems

    ds_v, pv, gv, sv = run("val")
    best = best_f1_threshold(pv, gv, ck["nc"])
    thr = best["thr"]
    print(f"Umbral de confianza elegido en VAL (máx. F1): {thr:.2f}  (F1 val = {best['f1']:.3f})")
    json.dump(best | {"tp": None, "fp": None, "fn": None}, open(res_dir / f"{args.name}_val_threshold.json", "w"), indent=2)

    # velocidad de inferencia: imagen por imagen, sin contar la carga de datos
    xs = [ds_v[i]["img"][None].to(dev).float() / 255 for i in range(min(110, len(ds_v)))]
    with torch.no_grad():
        for x in xs[:10]:
            decode(model(x))
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for x in xs[10:]:
            decode(model(x))
        if dev == "cuda":
            torch.cuda.synchronize()
    ms = (time.time() - t0) / max(len(xs) - 10, 1) * 1000
    summ = json.load(open(out / "runs" / "cnn" / args.name / "summary.json"))
    extra = dict(model=args.name, params_M=summ["params"] / 1e6, train_time_min=summ["train_time_min"],
                 best_epoch=summ["best_epoch"], inference_ms_per_image=ms, fps=1000 / ms)

    splits = {"val": (ds_v, pv, gv, sv)}
    if args.split == "test":
        splits["test"] = run("test")
    for sp, (ds, p, g, st) in splits.items():
        s, tab = full_report(f"{args.name}_{sp}", p, g, names, thr, res_dir, extra)
        save_predictions(pred_dir / f"{args.name}_{sp}.csv", st, p, names)
        draw_predictions(ds.files, st, p, g, names, max(thr, 0.3), fig_dir / f"{args.name}_{sp}_preds.jpg")
        print(f"\n=== {args.name} en {sp.upper()} ({s['n_images']} imágenes, {s['n_objects']} objetos, "
              f"{s['n_classes_evaluated']} clases evaluadas) ===")
        print(f"mAP@0.5 = {s['map50']:.3f} | mAP@0.5:0.95 = {s['map50_95']:.3f} | mAP@0.75 = {s['map75']:.3f}")
        print(f"Precisión = {s['precision']:.3f} | Recall = {s['recall']:.3f} | F1 = {s['f1']:.3f}  (umbral {thr:.2f})")
        print(tab.round(3).to_string(index=False))
    print(f"\nInferencia: {ms:.1f} ms por imagen ({1000 / ms:.0f} FPS) en {dev}")
    print("Archivos en results/eval/, results/preds/ y figures/")


if __name__ == "__main__":
    main()
