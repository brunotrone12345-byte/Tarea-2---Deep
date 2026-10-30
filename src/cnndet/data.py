"""
data.py - Dataset de detección (formato YOLO) con aumento de datos y targets tipo CenterNet.

Cada imagen se coloca en un lienzo cuadrado de img_size x img_size (letterbox, sin deformar).
En entrenamiento se aplica: escala aleatoria, traslación, volteo horizontal y jitter HSV.
En val/test no hay aumento: solo letterbox centrado.

Para cada imagen se devuelven los "targets" que usa la pérdida:
  hm   (nc, H/4, W/4)  mapa de calor: gaussiana en el centro de cada objeto, en el canal de su clase
  ind  (MAX_OBJ,)      índice de la celda del centro (fila*ancho + columna)
  mask (MAX_OBJ,)      1 si la posición corresponde a un objeto real
  wh   (MAX_OBJ,2)     ancho y alto de la caja en celdas de la salida
  off  (MAX_OBJ,2)     desplazamiento subcelda del centro
y, para evaluar, las cajas originales (gt_boxes, en píxeles de la imagen) y cómo volver a ellas (meta).
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

STRIDE = 4      # la salida de la red es 4 veces más pequeña que la entrada
MAX_OBJ = 64    # el máximo real en el dataset es 27 cajas por imagen


def load_data_cfg(yaml_path):
    cfg = yaml.safe_load(open(yaml_path, encoding="utf-8"))
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    return Path(cfg["path"]), names


def augment_hsv(img, hgain=0.015, sgain=0.6, vgain=0.4):
    """Cambia tono, saturación y brillo al azar (útil: el color del agua varía mucho)."""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def place(img, S, x0, y0):
    """Pega img en un lienzo gris SxS con esquina superior izquierda en (x0, y0); recorta lo que sobra."""
    canvas = np.full((S, S, 3), 114, np.uint8)
    nh, nw = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + nw, S), min(y0 + nh, S)
    if cx1 > cx0 and cy1 > cy0:
        canvas[cy0:cy1, cx0:cx1] = img[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return canvas


def build_targets(boxes, cls, nc, out):
    """boxes: (n,4) xyxy en píxeles del lienzo. Devuelve los arrays de targets."""
    hm = np.zeros((nc, out, out), np.float32)
    ind = np.zeros(MAX_OBJ, np.int64)
    mask = np.zeros(MAX_OBJ, np.float32)
    wh = np.ones((MAX_OBJ, 2), np.float32)
    off = np.zeros((MAX_OBJ, 2), np.float32)
    for k, (b, c) in enumerate(zip(boxes[:MAX_OBJ], cls[:MAX_OBJ])):
        x1, y1, x2, y2 = b / STRIDE
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ci, cj = min(int(cx), out - 1), min(int(cy), out - 1)
        sx, sy = max(0.54 * w / 6, 0.6), max(0.54 * h / 6, 0.6)   # sigma de la gaussiana
        rx, ry = max(int(3 * sx), 1), max(int(3 * sy), 1)
        xa, xb = max(ci - rx, 0), min(ci + rx + 1, out)
        ya, yb = max(cj - ry, 0), min(cj + ry + 1, out)
        xs, ys = np.arange(xa, xb) - ci, np.arange(ya, yb) - cj
        g = np.exp(-(xs[None, :] ** 2 / (2 * sx ** 2) + ys[:, None] ** 2 / (2 * sy ** 2)))
        hm[c, ya:yb, xa:xb] = np.maximum(hm[c, ya:yb, xa:xb], g)
        ind[k], mask[k] = cj * out + ci, 1.0
        wh[k], off[k] = (w, h), (cx - ci, cy - cj)
    return hm, ind, mask, wh, off


class FjordDataset(Dataset):
    def __init__(self, yaml_path, split, train=False, img_size=640):
        assert img_size % 32 == 0, "img_size debe ser múltiplo de 32"
        self.root, self.names = load_data_cfg(yaml_path)
        self.nc, self.train, self.S = len(self.names), train, img_size
        self.out = img_size // STRIDE
        self.files = sorted((self.root / "images" / split).glob("*.jpg"))
        self.labels = [self.root / "labels" / split / f"{f.stem}.txt" for f in self.files]
        assert self.files, f"No hay imágenes en {self.root / 'images' / split}"

    def __len__(self):
        return len(self.files)

    @staticmethod
    def read_labels(path):
        rows = [l.split() for l in path.read_text().splitlines() if l.strip()]
        return np.array(rows, np.float32).reshape(-1, 5)

    def __getitem__(self, i):
        img = cv2.imread(str(self.files[i]))
        h0, w0 = img.shape[:2]
        lab = self.read_labels(self.labels[i])
        cls = lab[:, 0].astype(np.int64)
        xyxy = np.stack([(lab[:, 1] - lab[:, 3] / 2) * w0, (lab[:, 2] - lab[:, 4] / 2) * h0,
                         (lab[:, 1] + lab[:, 3] / 2) * w0, (lab[:, 2] + lab[:, 4] / 2) * h0], 1)
        gt_boxes, gt_cls = xyxy.copy(), cls.copy()

        S = self.S
        s = S / max(h0, w0)
        dx = dy = 0.0
        if self.train:
            s *= float(np.exp(np.random.uniform(np.log(0.6), np.log(1.6))))  # escala log-uniforme
            dx, dy = np.random.uniform(-0.15, 0.15, 2) * S
        nw, nh = max(round(w0 * s), 1), max(round(h0 * s), 1)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        x0, y0 = int(round(S / 2 - nw / 2 + dx)), int(round(S / 2 - nh / 2 + dy))
        sx, sy = nw / w0, nh / h0
        canvas = place(img, S, x0, y0)

        b = xyxy.copy()
        b[:, [0, 2]] = b[:, [0, 2]] * sx + x0
        b[:, [1, 3]] = b[:, [1, 3]] * sy + y0
        area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        b = np.clip(b, 0, S)
        area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = (area1 > 0.4 * np.maximum(area0, 1e-6)) & ((b[:, 2] - b[:, 0]) > 2) & ((b[:, 3] - b[:, 1]) > 2)
        b, c = b[keep], cls[keep]

        if self.train:
            canvas = augment_hsv(canvas)
            if random.random() < 0.5:   # volteo horizontal
                canvas = canvas[:, ::-1]
                b = b.copy()
                b[:, [0, 2]] = S - b[:, [2, 0]]

        hm, ind, mask, wh, off = build_targets(b, c, self.nc, self.out)
        rgb = np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1))   # BGR->RGB, CHW (uint8)
        return dict(img=torch.from_numpy(rgb), hm=torch.from_numpy(hm), ind=torch.from_numpy(ind),
                    mask=torch.from_numpy(mask), wh=torch.from_numpy(wh), off=torch.from_numpy(off),
                    meta=torch.tensor([sx, sy, x0, y0, w0, h0], dtype=torch.float32),
                    gt_boxes=torch.from_numpy(gt_boxes.astype(np.float32)), gt_cls=torch.from_numpy(gt_cls),
                    name=self.files[i].stem, canvas_boxes=torch.from_numpy(b.astype(np.float32)),
                    canvas_cls=torch.from_numpy(c))


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("img", "hm", "ind", "mask", "wh", "off", "meta")}
    for k in ("gt_boxes", "gt_cls", "name", "canvas_boxes", "canvas_cls"):
        out[k] = [b[k] for b in batch]
    return out
