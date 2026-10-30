"""
metrics.py - Evaluación COMÚN para los cuatro modelos (mismo código = comparación justa).

Formato de entrada: una lista con un diccionario por imagen, en píxeles de la imagen original
  preds[i] = {"boxes": (n,4) xyxy, "scores": (n,), "labels": (n,)}
  gts[i]   = {"boxes": (m,4) xyxy, "labels": (m,)}

  coco_map          -> mAP@0.5 y mAP@0.5:0.95 (estándar COCO, vía torchmetrics/pycocotools)
  pr_at_threshold   -> precisión y recall (IoU>=0.5) para un umbral de confianza dado
  best_f1_threshold -> umbral de confianza que maximiza el F1 (se elige en VAL, nunca en test)

Solo se promedian las clases que tienen objetos en el conjunto evaluado.
"""
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou


def _tensors(items, with_scores):
    out = []
    for d in items:
        t = dict(boxes=torch.as_tensor(np.asarray(d["boxes"]), dtype=torch.float32).reshape(-1, 4),
                 labels=torch.as_tensor(np.asarray(d["labels"]), dtype=torch.int64).reshape(-1))
        if with_scores:
            t["scores"] = torch.as_tensor(np.asarray(d["scores"]), dtype=torch.float32).reshape(-1)
        out.append(t)
    return out


def coco_map(preds, gts, names):
    P, G = _tensors(preds, True), _tensors(gts, False)
    m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    m50 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)
    m.update(P, G)
    m50.update(P, G)
    r, r50 = m.compute(), m50.compute()
    per_class = {}
    for c, ap, ap50 in zip(r["classes"].tolist(), r["map_per_class"].tolist(), r50["map_per_class"].tolist()):
        if ap >= 0:   # -1 = clase sin objetos en el conjunto evaluado
            per_class[names[c]] = dict(ap50=ap50, ap50_95=ap)
    return dict(map50=float(r["map_50"]), map=float(r["map"]), map75=float(r["map_75"]),
                mar100=float(r["mar_100"]), per_class=per_class)


def pr_at_threshold(preds, gts, thr, nc, iou_thr=0.5):
    tp, fp, fn = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    for p, g in zip(preds, gts):
        keep = np.asarray(p["scores"]) >= thr
        pb = torch.as_tensor(np.asarray(p["boxes"]), dtype=torch.float32).reshape(-1, 4)[keep]
        ps = torch.as_tensor(np.asarray(p["scores"]), dtype=torch.float32)[keep]
        pl = torch.as_tensor(np.asarray(p["labels"]), dtype=torch.int64)[keep]
        gb = torch.as_tensor(np.asarray(g["boxes"]), dtype=torch.float32).reshape(-1, 4)
        gl = torch.as_tensor(np.asarray(g["labels"]), dtype=torch.int64)
        for c in set(gl.tolist()) | set(pl.tolist()):
            pc, sc, gc = pb[pl == c], ps[pl == c], gb[gl == c]
            pc = pc[sc.argsort(descending=True)]
            if len(gc) == 0:
                fp[c] += len(pc)
                continue
            if len(pc) == 0:
                fn[c] += len(gc)
                continue
            iou = box_iou(pc, gc).numpy()
            used = np.zeros(len(gc), bool)
            for i in range(len(pc)):
                j = int(np.where(~used, iou[i], -1).argmax())   # mejor objeto aún sin asignar
                if not used[j] and iou[i, j] >= iou_thr:
                    used[j] = True
                    tp[c] += 1
                else:
                    fp[c] += 1
            fn[c] += int((~used).sum())
    has = (tp + fn) > 0
    P_c, R_c = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    P, R = float(P_c[has].mean()), float(R_c[has].mean())
    return dict(thr=float(thr), precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9),
                micro_precision=float(tp.sum() / max(tp.sum() + fp.sum(), 1)),
                micro_recall=float(tp.sum() / max(tp.sum() + fn.sum(), 1)),
                tp=tp.tolist(), fp=fp.tolist(), fn=fn.tolist())


def best_f1_threshold(preds, gts, nc, grid=None):
    grid = np.arange(0.05, 0.95, 0.05) if grid is None else grid
    res = [pr_at_threshold(preds, gts, t, nc) for t in grid]
    return max(res, key=lambda r: r["f1"])