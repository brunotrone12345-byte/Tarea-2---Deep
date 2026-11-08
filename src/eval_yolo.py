"""
eval_yolo.py - Evalúa un modelo YOLO (Ultralytics) con NUESTRO evaluador común (el mismo que la CNN).

Uso (desde la carpeta del proyecto):
    python src/eval_yolo.py --name yolo26s_scratch --split val
    python src/eval_yolo.py --name yolo26s_scratch --split test      # resultado final (usar UNA vez)

Las predicciones se hacen con conf=0.001 (para que el mAP vea toda la curva). El umbral para
precisión/recall se elige SIEMPRE en val (máx. F1) y se aplica tal cual al test.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from detkit.metrics import best_f1_threshold
from detkit.report import draw_predictions, full_report, save_predictions


def read_gt(label_path, w, h):
    rows = [l.split() for l in Path(label_path).read_text().splitlines() if l.strip()]
    a = np.array(rows, np.float32).reshape(-1, 5)
    boxes = np.stack([(a[:, 1] - a[:, 3] / 2) * w, (a[:, 2] - a[:, 4] / 2) * h,
                      (a[:, 1] + a[:, 3] / 2) * w, (a[:, 2] + a[:, 4] / 2) * h], 1)
    return dict(boxes=boxes, labels=a[:, 0].astype(np.int64))


def predict_split(model, files, labels_dir, imgsz, device, bs=16):
    """Predice por trozos de `bs` imágenes. OJO: si se pasa una lista de rutas, Ultralytics IGNORA
    el parámetro batch y procesa TODA la lista de una vez (se agota la memoria de la GPU)."""
    preds, gts, stems = [], [], []
    for i in range(0, len(files), bs):
        chunk = [str(f) for f in files[i:i + bs]]
        for r in model.predict(source=chunk, imgsz=imgsz, conf=0.001, max_det=300, verbose=False, device=device):
            h, w = r.orig_shape
            stem = Path(r.path).stem
            preds.append(dict(boxes=r.boxes.xyxy.cpu().numpy(), scores=r.boxes.conf.cpu().numpy(),
                              labels=r.boxes.cls.cpu().numpy().astype(np.int64)))
            gts.append(read_gt(Path(labels_dir) / f"{stem}.txt", w, h))
            stems.append(stem)
    return preds, gts, stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="nombre de la carpeta en runs/yolo/")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--yaml", default="data/fjord/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out = Path(args.out)
    res_dir, fig_dir, pred_dir = out / "results" / "eval", out / "figures", out / "results" / "preds"
    for d in (res_dir, fig_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    run = out / "runs" / "yolo" / args.name
    model = YOLO(run / "weights" / "best.pt")
    from detkit.paths import resolve_dataset_cfg
    root, names, _ = resolve_dataset_cfg(args.yaml)

    def files_of(split):
        return sorted((root / "images" / split).glob("*.jpg")), root / "labels" / split

    fv, lv = files_of("val")
    pv, gv, sv = predict_split(model, fv, lv, args.imgsz, args.device)
    best = best_f1_threshold(pv, gv, len(names))
    thr = best["thr"]
    print(f"Umbral de confianza elegido en VAL (máx. F1): {thr:.2f}  (F1 val = {best['f1']:.3f})")
    json.dump(best | {"tp": None, "fp": None, "fn": None}, open(res_dir / f"{args.name}_val_threshold.json", "w"), indent=2)

    # velocidad: UNA imagen por llamada (batch 1). Incluye preprocesado + inferencia + posprocesado,
    # no la lectura de disco
    sp = []
    for i, f in enumerate(fv[:110]):
        r = model.predict(source=str(f), imgsz=args.imgsz, conf=0.001, verbose=False, device=args.device)[0]
        if i >= 10:   # las 10 primeras son de calentamiento
            sp.append(sum(r.speed.values()))
    ms = float(np.mean(sp))
    summ = json.load(open(run / "summary.json"))
    extra = dict(model=args.name, params_M=summ["params"] / 1e6, train_time_min=summ["total_time_min"],
                 best_epoch=summ["best_epoch"], inference_ms_per_image=ms, fps=1000 / ms)

    splits = {"val": (fv, pv, gv, sv)}
    if args.split == "test":
        ft, lt = files_of("test")
        splits["test"] = (ft, *predict_split(model, ft, lt, args.imgsz, args.device))
    for sp_name, (files, p, g, st) in splits.items():
        s, tab = full_report(f"{args.name}_{sp_name}", p, g, names, thr, res_dir, extra)
        save_predictions(pred_dir / f"{args.name}_{sp_name}.csv", st, p, names)
        draw_predictions(files, st, p, g, names, max(thr, 0.3), fig_dir / f"{args.name}_{sp_name}_preds.jpg")
        print(f"\n=== {args.name} en {sp_name.upper()} ({s['n_images']} imágenes, {s['n_objects']} objetos, "
              f"{s['n_classes_evaluated']} clases evaluadas) ===")
        print(f"mAP@0.5 = {s['map50']:.3f} | mAP@0.5:0.95 = {s['map50_95']:.3f} | mAP@0.75 = {s['map75']:.3f}")
        print(f"Precisión = {s['precision']:.3f} | Recall = {s['recall']:.3f} | F1 = {s['f1']:.3f}  (umbral {thr:.2f})")
        print(tab.round(3).to_string(index=False))
    print(f"\nInferencia: {ms:.1f} ms por imagen ({1000 / ms:.0f} FPS)")
    print("Archivos en results/eval/, results/preds/ y figures/")


if __name__ == "__main__":
    main()