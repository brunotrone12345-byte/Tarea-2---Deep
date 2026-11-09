"""Evalúa Faster R-CNN con el mismo detkit usado por CNN y YOLO.

Desde la raíz del proyecto (con .venv activo):
  python src/eval_frcnn.py --name frcnn_transfer_final --split val --device cuda
  python src/eval_frcnn.py --name frcnn_transfer_final --split test --device cuda

El umbral de F1 se selecciona exclusivamente en VAL y se aplica sin cambios a TEST.
No entrena ni modifica los pesos del modelo.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from train_frcnn import build_model
from detkit.metrics import best_f1_threshold
from detkit.report import draw_predictions, full_report, save_predictions


class FjordEvalDataset(Dataset):
    """Devuelve imágenes originales y cajas xyxy en píxeles; clases 0..nc-1."""

    def __init__(self, yaml_path, split):
        if split not in ("val", "test"):
            raise ValueError("Solo se admiten val y test")

        cfg_path = Path(yaml_path).resolve()
        with cfg_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        root = Path(cfg["path"])
        if not root.is_absolute():
            root = (cfg_path.parent / root).resolve()

        self.nc = int(cfg["nc"])
        names = cfg["names"]
        self.names = (
            [names[i] if i in names else names[str(i)] for i in range(self.nc)]
            if isinstance(names, dict) else list(names)
        )
        self.files = sorted((root / "images" / split).glob("*.jpg"))
        self.labels_dir = root / "labels" / split
        if not self.files or not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Faltan imágenes o etiquetas de {split} en {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image_path = self.files[i]
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            x = to_tensor(im)

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Falta la etiqueta: {label_path}")

        boxes, labels = [], []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Etiqueta no válida en {label_path}: {line!r}")
            c = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            if not 0 <= c < self.nc:
                raise ValueError(f"Clase fuera de rango en {label_path}: {c}")
            x1 = max(0.0, min(float(w), (xc - bw / 2) * w))
            y1 = max(0.0, min(float(h), (yc - bh / 2) * h))
            x2 = max(0.0, min(float(w), (xc + bw / 2) * w))
            y2 = max(0.0, min(float(h), (yc + bh / 2) * h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(c)

        gt = {
            "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(labels, dtype=np.int64),
        }
        return x, gt, image_path.stem


@torch.inference_mode()
def predict_split(model, ds, device, measure_speed=False):
    preds, gts, stems, elapsed_ms = [], [], [], []
    for i in range(len(ds)):
        image, gt, stem = ds[i]
        x = image.to(device)
        if measure_speed and i < 110 and device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = model([x])[0]
        if measure_speed and i < 110 and device.type == "cuda":
            torch.cuda.synchronize()
        if measure_speed and 10 <= i < 110:
            elapsed_ms.append((time.perf_counter() - start) * 1000)

        # Torchvision reserva la clase 0 para fondo; detkit usa 0..27.
        labels = output["labels"].detach().cpu().numpy().astype(np.int64) - 1
        valid = (labels >= 0) & (labels < ds.nc)
        preds.append({
            "boxes": output["boxes"].detach().cpu().numpy()[valid],
            "scores": output["scores"].detach().cpu().numpy()[valid],
            "labels": labels[valid],
        })
        gts.append(gt)
        stems.append(stem)

    mean_ms = float(np.mean(elapsed_ms)) if elapsed_ms else None
    return preds, gts, stems, mean_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Nombre de runs/frcnn/<name>")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--yaml", default="data/fjord/data.yaml")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--out", default=".", help="Raíz del proyecto")
    ap.add_argument("--score-thr", type=float, default=0.001)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA no está disponible")
    if not (0 <= args.score_thr <= 1):
        ap.error("--score-thr debe estar entre 0 y 1")

    root = Path(args.out)
    run = root / "runs" / "frcnn" / args.name
    ck_path = run / "weights" / "best.pt"
    if not ck_path.is_file():
        ap.error(f"No existe el checkpoint: {ck_path}")

    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    ds_val = FjordEvalDataset(root / args.yaml, "val")
    if ds_val.nc != int(ck["nc"]) or ds_val.names != list(ck["names"]):
        ap.error("Las clases del dataset no coinciden con las del checkpoint")

    model = build_model(
        int(ck["nc"]), int(ck["imgsz"]), int(ck["max_size"]),
        int(ck["trainable_backbone_layers"]), pretrained=False
    )
    model.load_state_dict(ck["model"], strict=True)
    # Las detecciones de baja confianza son necesarias para el cálculo del mAP.
    model.roi_heads.score_thresh = args.score_thr
    device = torch.device(args.device)
    model = model.to(device).eval()

    res_dir = root / "results" / "eval"
    pred_dir = root / "results" / "preds"
    fig_dir = root / "figures"
    for folder in (res_dir, pred_dir, fig_dir):
        folder.mkdir(parents=True, exist_ok=True)

    print(f"Modelo: {args.name} | mejor época: {ck['epoch']} | VAL: {len(ds_val)} imágenes")
    pv, gv, sv, ms = predict_split(model, ds_val, device, measure_speed=True)
    best = best_f1_threshold(pv, gv, ds_val.nc)
    thr = float(best["thr"])
    print(f"Umbral de confianza elegido en VAL (máx. F1): {thr:.2f} | F1 VAL: {best['f1']:.3f}")
    (res_dir / f"{args.name}_val_threshold.json").write_text(
        json.dumps({**best, "tp": None, "fp": None, "fn": None}, indent=2),
        encoding="utf-8"
    )

    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    extra = dict(
        model=args.name,
        params_M=sum(p.numel() for p in model.parameters()) / 1e6,
        train_time_min=summ["total_time_min"],  # incluye validaciones, igual que resumen original
        best_epoch=int(ck["epoch"]),
        inference_ms_per_image=ms,
        fps=1000 / ms if ms else None,
    )
    splits = {"val": (ds_val, pv, gv, sv)}
    if args.split == "test":
        ds_test = FjordEvalDataset(root / args.yaml, "test")
        if ds_test.nc != ds_val.nc or ds_test.names != ds_val.names:
            ap.error("Las clases de TEST difieren de VALIDACIÓN")
        pt, gt, st, _ = predict_split(model, ds_test, device)
        splits["test"] = (ds_test, pt, gt, st)

    for split, (ds, preds, gts, stems) in splits.items():
        tag = f"{args.name}_{split}"
        summary, table = full_report(tag, preds, gts, ds.names, thr, res_dir, extra)
        save_predictions(pred_dir / f"{tag}.csv", stems, preds, ds.names)
        draw_predictions(
            ds.files, stems, preds, gts, ds.names,
            max(thr, 0.3), fig_dir / f"{tag}_preds.jpg"
        )
        print(f"\n=== {args.name} en {split.upper()} ")
        print(f"Imágenes: {summary['n_images']} | Objetos: {summary['n_objects']} | Clases: {summary['n_classes_evaluated']}")
        print(f"mAP@0.5={summary['map50']:.3f} | mAP@0.5:0.95={summary['map50_95']:.3f} | mAP@0.75={summary['map75']:.3f}")
        print(f"Precision={summary['precision']:.3f} | Recall={summary['recall']:.3f} | F1={summary['f1']:.3f} (umbral {thr:.2f})")
        print(table.round(3).to_string(index=False))

    if ms:
        print(f"\nInferencia VAL: {ms:.1f} ms/imagen ({1000 / ms:.1f} FPS) en {device}")
    print("Archivos guardados en results/eval/, results/preds/ y figures/")


if __name__ == "__main__":
    main()
