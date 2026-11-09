"""Entrenamiento de Faster R-CNN ResNet50-FPN con pesos COCO para Fjord.

Ejecutar desde la RAIZ del proyecto (entorno .venv activado):
  python src/train_frcnn.py --name frcnn_prueba --epochs 1 --limit-train 16 --limit-val 12 --imgsz 320 --max-size 512 --batch 1 --workers 0 --device cuda

La prueba es solo para comprobar compatibilidad con GPU. Para el entrenamiento real,
no usar limites de train/val. No se utiliza el split TEST en este script.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from detkit.metrics import best_f1_threshold

class FjordDetectionDataset(Dataset):
    """Convierte anotaciones YOLO (clases 0..27) al formato Torchvision (1..28)."""

    def __init__(self, yaml_path: str, split: str):
        if split not in ("train", "val"):
            raise ValueError("Este script solo admite train y val; no se toca TEST")
        cfg_path = Path(yaml_path).resolve()
        with cfg_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.root = Path(cfg["path"])
        if not self.root.is_absolute():
            self.root = (cfg_path.parent / self.root).resolve()
        self.nc = int(cfg["nc"])
        names = cfg["names"]
        self.names = [names[i] if i in names else names[str(i)] for i in range(self.nc)] if isinstance(names, dict) else list(names)
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.files = sorted(self.image_dir.glob("*.jpg"))
        if not self.files:
            raise FileNotFoundError(f"No hay imágenes JPG en {self.image_dir}. Revisa data/fjord/data.yaml")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"Falta carpeta de etiquetas: {self.label_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fp = self.files[i]
        with Image.open(fp) as im:
            im = im.convert("RGB")
            w, h = im.size
            img = to_tensor(im)

        lab_path = self.label_dir / (fp.stem + ".txt")
        if not lab_path.exists():
            raise FileNotFoundError(f"No se encontró la etiqueta de {fp.name}: {lab_path}")
        boxes, labels = [], []
        for line_no, line in enumerate(lab_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Etiqueta no válida: {lab_path}:{line_no} -> {line!r}")
            c = int(parts[0]); xc, yc, bw, bh = map(float, parts[1:])
            if not (0 <= c < self.nc):
                raise ValueError(f"Clase fuera de rango ({c}) en {lab_path}:{line_no}")
            x1 = max(0.0, min(float(w), (xc - bw / 2) * w))
            y1 = max(0.0, min(float(h), (yc - bh / 2) * h))
            x2 = max(0.0, min(float(w), (xc + bw / 2) * w))
            y2 = max(0.0, min(float(h), (yc + bh / 2) * h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(c + 1) 
        b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        l = torch.tensor(labels, dtype=torch.int64)
        target = {
            "boxes": b,
            "labels": l,
            "image_id": torch.tensor([i], dtype=torch.int64),
            "area": (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]),
            "iscrowd": torch.zeros(len(l), dtype=torch.int64),
        }
        return img, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(nc, imgsz, max_size, trainable_backbone_layers, pretrained):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers if pretrained else None,
        min_size=imgsz,
        max_size=max_size,
        rpn_pre_nms_top_n_train=500,
        rpn_post_nms_top_n_train=250,
        rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_test=250,
        box_batch_size_per_image=128,
        box_detections_per_img=100,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc + 1)
    return model


def evaluate(model, loader, device, nc):
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox"
    )

    preds, gts = [], []
    with torch.inference_mode():
        for images, targets in tqdm(
            loader, desc="Validacion", leave=False
        ):
            inp = [im.to(device) for im in images]
            outputs = model(inp)

            metric.update(
                [
                    {k: v.detach().cpu() for k, v in o.items()}
                    for o in outputs
                ],
                [
                    {"boxes": t["boxes"], "labels": t["labels"]}
                    for t in targets
                ]
            )

            for o, t in zip(outputs, targets):
                labels = (
                    o["labels"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype("int64") - 1
                )

                valid = (labels >= 0) & (labels < nc)

                preds.append({
                    "boxes": o["boxes"].detach().cpu().numpy()[valid],
                    "scores": o["scores"].detach().cpu().numpy()[valid],
                    "labels": labels[valid]
                })

                gts.append({
                    "boxes": t["boxes"].cpu().numpy(),
                    "labels": t["labels"].cpu().numpy() - 1
                })

    m = metric.compute()

    pr = best_f1_threshold(preds, gts, nc)

    return {
        "map50": float(m["map_50"]),
        "map50_95": float(m["map"]),
        "map75": float(m["map_75"]),
        "mar100": float(m["mar_100"]),
        "precision": float(pr["precision"]),
        "recall": float(pr["recall"]),
        "f1": float(pr["f1"]),
        "conf_threshold": float(pr["thr"])
    }

def save_checkpoint_atomic(ck, path):
    path = Path(path)
    temporal = path.with_name(path.name + ".tmp")
    torch.save(ck, temporal)
    temporal.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="data/fjord/data.yaml")
    ap.add_argument("--name", default="frcnn_transfer_prueba")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=320, help="lado menor en preprocesamiento")
    ap.add_argument("--max-size", type=int, default=512, help="lado mayor maximo")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=16, help="0 = todas las imágenes")
    ap.add_argument("--limit-val", type=int, default=12, help="0 = todas las imágenes")
    ap.add_argument("--trainable-backbone-layers", type=int, default=0, choices=range(6))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=0,
                    help="0 desactiva el early stopping")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    if args.patience < 0:
        ap.error("--patience debe ser 0 o mayor")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.max_size < args.imgsz:
        ap.error("epochs, batch, imgsz y max-size tienen valores incompatibles")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA no esta disponible. Ejecuta python check_env.py")
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train_ds = FjordDetectionDataset(args.yaml, "train")
    val_ds = FjordDetectionDataset(args.yaml, "val")
    if train_ds.nc != val_ds.nc:
        raise ValueError("Numero de clases inconsistente en los splits")
    ntrain_full, nval_full = len(train_ds), len(val_ds)
    if args.limit_train > 0:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val > 0:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))
    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate)
    va = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate)

    out = Path("runs") / "frcnn" / args.name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"La carpeta {out} ya tiene resultados. Usa otro --name para no sobrescribirlos")
    (out / "weights").mkdir(parents=True)
    print(f"GPU/CPU: {device}; TRAIN: {len(train_ds)}/{ntrain_full}; VAL: {len(val_ds)}/{nval_full}")
    print(f"Clases: {train_ds.dataset.nc if isinstance(train_ds, Subset) else train_ds.nc} de datos + fondo = 29 en Faster R-CNN")
    print("Cargando Faster R-CNN ResNet50-FPN con pesos COCO (puede descargarlos)...")
    model = build_model(train_ds.dataset.nc if isinstance(train_ds, Subset) else train_ds.nc,
                        args.imgsz, args.max_size, args.trainable_backbone_layers, pretrained=True).to(device)
    print(f"Parametros totales: {sum(p.numel() for p in model.parameters())/1e6:.2f}M; entrenables: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    cfg = vars(args).copy(); cfg["total_train"] = ntrain_full; cfg["total_val"] = nval_full
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    hist = []
    best_map = -1.0
    best_epoch = 0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        model.train()
        loss_sum = 0.0

        loss_parts = {
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0
        }
        for images, targets in tqdm(tr, desc=f"Epoca {epoch}/{args.epochs}"):
            xs = [im.to(device) for im in images]
            ys = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(xs, ys)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Perdida no finita: {losses}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0)
            opt.step()
            loss_sum += loss.item()
            for k in loss_parts:
                loss_parts[k] += losses[k].item()
        metrics = evaluate(model, va, device, train_ds.dataset.nc if isinstance(train_ds, Subset) else train_ds.nc)
        avg_loss = loss_sum / max(len(tr), 1)
        elapsed_min = (time.perf_counter() - t_start)/60
        print(f"Epoca {epoch}/{args.epochs} | loss={avg_loss:.4f} | "
              f"val_mAP50={metrics['map50']:.4f} | val_mAP50-95={metrics['map50_95']:.4f} | "
              f"tiempo={elapsed_min:.1f} min")
        row = {
            "epoch": epoch,
            "loss": avg_loss,
            **{
                k: v / max(len(tr), 1)
                for k, v in loss_parts.items()
            },
            **metrics,
            "epoch_time_min": (time.perf_counter() - epoch_start) / 60,
            "elapsed_min": elapsed_min
        }
        hist.append(row)
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(hist[0]))
            writer.writeheader()
            writer.writerows(hist)
            
        improved = metrics["map50_95"] > best_map

        if improved:
            best_map = metrics["map50_95"]
            best_epoch = epoch

        ck = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "nc": train_ds.dataset.nc if isinstance(train_ds, Subset) else train_ds.nc,
            "names": train_ds.dataset.names if isinstance(train_ds, Subset) else train_ds.names,
            "imgsz": args.imgsz,
            "max_size": args.max_size,
            "trainable_backbone_layers": args.trainable_backbone_layers,
            "best_map": best_map,
            "best_epoch": best_epoch,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "val_metrics": metrics
        }
        save_checkpoint_atomic(
            ck,
            out / "weights" / "last.pt"
        )
        if improved:
            save_checkpoint_atomic(
                ck,
                out / "weights" / "best.pt"
            )

            print(f"  -> Nuevo mejor checkpoint: epoca {epoch}")
        if args.patience > 0 and epoch - best_epoch >= args.patience:
            print(f"Early stopping: {args.patience} epocas sin mejorar.")
            break
    summ = {"name": args.name, "model": "fasterrcnn_resnet50_fpn", "mode": "transfer_coco",
            "epochs_requested": args.epochs, "epochs_run": len(hist), "best_epoch": best_epoch,
            "val_map50_95_at_best": best_map, "total_time_min": (time.perf_counter() - t_start)/60,
            "train_images": len(train_ds), "val_images": len(val_ds), "args": cfg}
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("\n=== RESUMEN ===")
    print(json.dumps(summ, indent=2))
    print(f"Pesos en {out / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
