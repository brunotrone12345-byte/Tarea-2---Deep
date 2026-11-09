"""
report.py - Informe de resultados COMÚN para los cuatro modelos.

  save_predictions  -> CSV con todas las detecciones (se entrega y permite recalcular métricas)
  full_report       -> metrics.json + per_class.csv + gráfico de AP por clase
  draw_predictions  -> cuadrícula de imágenes con cajas reales (verde) y predichas (rojo)
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import coco_map, pr_at_threshold


def save_predictions(path, image_names, preds, class_names):
    rows = []
    for name, p in zip(image_names, preds):
        for b, s, l in zip(p["boxes"], p["scores"], p["labels"]):
            rows.append((name, *[round(float(v), 2) for v in b], round(float(s), 4), int(l), class_names[int(l)]))
    pd.DataFrame(rows, columns=["image", "x1", "y1", "x2", "y2", "score", "class_id", "class"]).to_csv(path, index=False)


def full_report(tag, preds, gts, class_names, thr, out_dir, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nc = len(class_names)
    m, pr = coco_map(preds, gts, class_names), pr_at_threshold(preds, gts, thr, nc)
    rows = []
    for c, name in enumerate(class_names):
        tp, fp, fn = pr["tp"][c], pr["fp"][c], pr["fn"][c]
        if tp + fn == 0:
            continue   # clase sin objetos en este conjunto: no se evalúa
        ap = m["per_class"].get(name, {})
        rows.append(dict(class_id=c, clase=name, n_gt=int(tp + fn), ap50=ap.get("ap50"), ap50_95=ap.get("ap50_95"),
                         precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1)))
    tab = pd.DataFrame(rows).sort_values("n_gt", ascending=False)
    tab.to_csv(out_dir / f"{tag}_per_class.csv", index=False)
    summary = dict(tag=tag, n_images=len(gts), n_objects=int(sum(len(g["labels"]) for g in gts)),
                   n_classes_evaluated=len(tab), conf_threshold=thr, map50=m["map50"], map50_95=m["map"],
                   map75=m["map75"], precision=pr["precision"], recall=pr["recall"], f1=pr["f1"],
                   micro_precision=pr["micro_precision"], micro_recall=pr["micro_recall"])
    summary.update(extra or {})
    json.dump(summary, open(out_dir / f"{tag}_metrics.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(tab) + 1.5))
    y = np.arange(len(tab))
    ax.barh(y + 0.2, tab.ap50.fillna(0), 0.4, label="AP@0.5")
    ax.barh(y - 0.2, tab.ap50_95.fillna(0), 0.4, label="AP@0.5:0.95")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(tab.clase, tab.n_gt)])
    ax.invert_yaxis()
    ax.set_xlabel("AP")
    ax.set_title(f"{tag}: AP por clase")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_per_class_ap.png", dpi=150)
    plt.close()
    return summary, tab


def draw_predictions(files, image_names, preds, gts, class_names, thr, out_path, n=12, seed=0, cols=3, w=640):
    """Muestra n imágenes ELEGIDAS AL AZAR (no a dedo): real = verde, predicción = rojo."""
    by_name = {Path(f).stem: f for f in files}
    idx = np.random.default_rng(seed).choice(len(image_names), size=min(n, len(image_names)), replace=False)
    tiles = []
    for i in idx:
        im = cv2.imread(str(by_name[image_names[i]]))
        k = w / im.shape[1]
        im = cv2.resize(im, (w, round(im.shape[0] * k)))
        for b, l in zip(gts[i]["boxes"], gts[i]["labels"]):
            x1, y1, x2, y2 = (b * k).astype(int)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for b, s, l in zip(preds[i]["boxes"], preds[i]["scores"], preds[i]["labels"]):
            if s < thr:
                continue
            x1, y1, x2, y2 = (b * k).astype(int)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(im, f"{class_names[int(l)]} {s:.2f}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(im)
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        h = max(t.shape[0] for t in row)
        row = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in row]
        row += [np.zeros((h, w, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(str(out_path), np.vstack(rows))
