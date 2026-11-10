"""
robustness_eval.py - Etapa 7: robustez del mejor modelo frente a degradaciones de las imágenes de test.

Se generan, en memoria y de forma determinista (semilla fija), versiones degradadas de las 451 imágenes de test:
    brillo      reducción de iluminación (multiplica la intensidad por k)
    contraste   disminución del contraste (acerca cada pixel a la media de la imagen)
    desenfoque  desenfoque gaussiano (sigma proporcional al tamaño de la imagen)
    ruido       ruido gaussiano aditivo
    neblina     turbidez: mezcla con un velo verdoso (típico del agua)
con 3 niveles cada una (leve, medio, fuerte). Se evalúa con las MISMAS métricas y el MISMO umbral de confianza
elegido en val con imágenes limpias (no se vuelve a ajustar: así se mide la degradación real).

Uso (desde la carpeta del proyecto):
    python src/robustness_eval.py --name yolo26s_tl_full
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from detkit.metrics import coco_map, pr_at_threshold
from detkit.paths import resolve_dataset_root
from eval_yolo import read_gt

ROOT = Path(".")
LEVELS = {  # tipo -> parámetros de los niveles 1 (leve), 2 (medio), 3 (fuerte)
    "brillo": [0.6, 0.35, 0.15],
    "contraste": [0.6, 0.35, 0.15],
    "desenfoque": [2, 5, 10],
    "ruido": [10, 25, 50],
    "neblina": [0.2, 0.4, 0.6],
}
VEIL_BGR = np.array([130, 160, 90], np.float32)   # velo verdoso (BGR)


def degrade(img, kind, level, rng):
    if kind == "desenfoque":
        sigma = level * max(img.shape[:2]) / 1280      # relativo al tamaño: las imágenes miden hasta 1280 px
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    f = img.astype(np.float32)
    if kind == "brillo":
        f = f * level
    elif kind == "contraste":
        m = f.mean()
        f = m + level * (f - m)
    elif kind == "ruido":
        f = f + rng.normal(0, level, f.shape).astype(np.float32)
    elif kind == "neblina":
        f = f * (1 - level) + VEIL_BGR * level
    return np.clip(f, 0, 255).astype(np.uint8)


def predict_arrays(model, imgs, imgsz, device, bs=16):
    """Predice por trozos (Ultralytics procesa TODA la lista de una vez si no se trocea)."""
    out = []
    for i in range(0, len(imgs), bs):
        for r in model.predict(source=imgs[i:i + bs], imgsz=imgsz, conf=0.001, max_det=300, verbose=False, device=device):
            out.append(dict(boxes=r.boxes.xyxy.cpu().numpy(), scores=r.boxes.conf.cpu().numpy(),
                            labels=r.boxes.cls.cpu().numpy().astype(np.int64)))
    return out


def args_example_index(args, files, base_preds, gts, thr):
    if args.example_stem:
        return [f.stem for f in files].index(args.example_stem)
    return pick_example(base_preds, gts, thr)


def pick_example(preds, gts, thr):
    """Imagen (con al menos 2 peces) donde el modelo acierta más objetos en la versión ORIGINAL: así se ve cómo se pierde."""
    best, k = -1, 0
    for i, (p, g) in enumerate(zip(preds, gts)):
        if len(g["labels"]) < 2:
            continue
        keep = p["scores"] >= thr
        n = 0
        for b, c in zip(g["boxes"], g["labels"]):
            m = keep & (p["labels"] == c)
            if m.any():
                pb = p["boxes"][m]
                iw = np.clip(np.minimum(b[2], pb[:, 2]) - np.maximum(b[0], pb[:, 0]), 0, None)
                ih = np.clip(np.minimum(b[3], pb[:, 3]) - np.maximum(b[1], pb[:, 1]), 0, None)
                inter = iw * ih
                iou = inter / ((b[2] - b[0]) * (b[3] - b[1]) + (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]) - inter + 1e-9)
                n += int((iou >= 0.5).any())
        if n > best:
            best, k = n, i
    return k


def make_examples(model, imgs, gts, k, thr, args, out_f):
    """Una imagen bajo cada condición (verde = objetos reales, rojo = predicciones con confianza >= umbral)."""
    tiles, W = [], 480
    for kind in ["original"] + list(LEVELS):
        row = []
        for lv in range(4):
            if (kind == "original") != (lv == 0):
                row.append(np.zeros((int(W * imgs[k].shape[0] / imgs[k].shape[1]), W, 3), np.uint8))
                continue
            ki = list(LEVELS).index(kind) if kind != "original" else 0
            im = imgs[k] if kind == "original" else degrade(imgs[k], kind, LEVELS[kind][lv - 1],
                                                            np.random.default_rng([args.seed, ki, lv, k]))
            p = predict_arrays(model, [im], args.imgsz, args.device)[0]
            s = W / im.shape[1]
            t = cv2.resize(im, (W, int(im.shape[0] * s)))
            for b in gts[k]["boxes"]:
                cv2.rectangle(t, tuple((b[:2] * s).astype(int)), tuple((b[2:] * s).astype(int)), (0, 200, 0), 2)
            for b, sc in zip(p["boxes"], p["scores"]):
                if sc >= thr:
                    cv2.rectangle(t, tuple((b[:2] * s).astype(int)), tuple((b[2:] * s).astype(int)), (0, 0, 255), 2)
            cv2.putText(t, kind if lv == 0 else f"{kind} {['leve', 'medio', 'fuerte'][lv - 1]}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            row.append(t)
        tiles.append(np.hstack(row))
    cv2.imwrite(str(out_f / "fig_examples.jpg"), np.vstack(tiles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="yolo26s_tl_full")
    ap.add_argument("--yaml", default="data/fjord/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples-only", type=int, default=0, help="1 = solo regenerar fig_examples.jpg (rápido)")
    ap.add_argument("--example-stem", default="", help="nombre de la imagen para los ejemplos (ej. 1234)")
    args = ap.parse_args()
    out_t, out_f = ROOT / "results" / "robustness", ROOT / "figures" / "robustness"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(args.yaml, encoding="utf-8"))
    names = [cfg["names"][k] for k in sorted(cfg["names"])] if isinstance(cfg["names"], dict) else cfg["names"]
    root = resolve_dataset_root(args.yaml, cfg["path"])
    files = sorted((root / "images" / "test").glob("*.jpg"))
    imgs = [cv2.imread(str(f)) for f in files]
    gts = [read_gt(root / "labels" / "test" / f"{f.stem}.txt", im.shape[1], im.shape[0])
           for f, im in zip(files, imgs)]
    thr = json.load(open(ROOT / "results" / "eval" / f"{args.name}_val_threshold.json"))["thr"]
    model = YOLO(ROOT / "runs" / "yolo" / args.name / "weights" / "best.pt")
    print(f"{len(imgs)} imágenes de test | umbral de confianza (fijado en val limpio): {thr:.2f}")

    if args.examples_only:
        base_preds = predict_arrays(model, imgs, args.imgsz, args.device)
        k = args_example_index(args, files, base_preds, gts, thr)
        make_examples(model, imgs, gts, k, thr, args, out_f)
        print(f"Ejemplos regenerados con la imagen '{files[k].stem}' -> {out_f / 'fig_examples.jpg'}")
        return

    conds = [("original", 0, 0.0)] + [(k, lv + 1, p) for k, ps in LEVELS.items() for lv, p in enumerate(ps)]
    rows = []
    for ci, (kind, lv, par) in enumerate(conds):
        if kind == "original":
            batch = imgs
        else:
            ki = list(LEVELS).index(kind)
            batch = [degrade(im, kind, par, np.random.default_rng([args.seed, ki, lv, j])) for j, im in enumerate(imgs)]
        preds = predict_arrays(model, batch, args.imgsz, args.device)
        if kind == "original":
            base_preds = preds
        m, pr = coco_map(preds, gts, names), pr_at_threshold(preds, gts, thr, len(names))
        rows.append(dict(tipo=kind, nivel=lv, parametro=par, map50=m["map50"], map50_95=m["map"], map75=m["map75"],
                         precision=pr["precision"], recall=pr["recall"], f1=pr["f1"]))
        print(f"  {kind:<10} nivel {lv} (param {par:<5}) mAP@0.5 {m['map50']:.3f}  mAP@0.5:0.95 {m['map']:.3f}  "
              f"P {pr['precision']:.3f}  R {pr['recall']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    base = df.iloc[0]
    for c in ("map50", "map50_95", "map75", "precision", "recall", "f1"):
        df[f"caida_rel_{c}_%"] = 100 * (1 - df[c] / base[c]) if base[c] > 0 else np.nan
    df.round(4).to_csv(out_t / "robustness_metrics.csv", index=False)

    ref = ROOT / "results" / "eval" / f"{args.name}_test_metrics.json"
    if ref.exists():
        r = json.load(open(ref))
        print(f"\nControl: imágenes originales -> mAP@0.5 {base.map50:.3f} / mAP@0.5:0.95 {base.map50_95:.3f}  "
              f"(evaluación anterior en test: {r['map50']:.3f} / {r['map50_95']:.3f})")

    print("\n=== Caída relativa del mAP@0.5:0.95 respecto a las imágenes originales (%) ===")
    piv = df[df.tipo != "original"].pivot(index="tipo", columns="nivel", values="caida_rel_map50_95_%").loc[list(LEVELS)]
    piv.columns = ["leve", "medio", "fuerte"]
    print(piv.round(1).to_string())

    # ---- figuras ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, (c, ttl) in zip(axs, [("map50", "mAP@0.5"), ("map50_95", "mAP@0.5:0.95")]):
        for kind in LEVELS:
            d = df[df.tipo == kind]
            ax.plot([0] + list(d.nivel), [base[c]] + list(d[c]), "o-", label=kind)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["original", "leve", "medio", "fuerte"])
        ax.set_ylabel(ttl + " en test")
        ax.grid(alpha=.3)
    axs[0].legend(fontsize=8)
    fig.suptitle(f"Robustez de {args.name} frente a degradaciones", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_f / "fig_robustness_curves.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    im_ = ax.imshow(piv.values, cmap="Reds", vmin=0, vmax=max(60, np.nanmax(piv.values) if np.isfinite(piv.values).any() else 60))
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if piv.values[i, j] > 35 else "black")
    ax.set_xticks(range(3))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index)
    ax.set_title("Caída relativa de mAP@0.5:0.95", fontsize=10)
    plt.colorbar(im_, ax=ax, fraction=0.05)
    plt.tight_layout()
    plt.savefig(out_f / "fig_robustness_heatmap.png", dpi=150)
    plt.close()

    k = args_example_index(args, files, base_preds, gts, thr)
    make_examples(model, imgs, gts, k, thr, args, out_f)
    print(f"\nListo: {out_t / 'robustness_metrics.csv'} y figuras en {out_f} (imagen de ejemplo: {files[k].stem})")


if __name__ == "__main__":
    main()