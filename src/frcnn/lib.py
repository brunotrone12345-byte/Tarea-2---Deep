"""
lib.py - Faster R-CNN (torchvision): dataset, modelo, predicción y pérdida de validación.

Faster R-CNN es un detector de DOS ETAPAS:
  1) RPN (Region Proposal Network): propone regiones donde "podría haber un objeto".
  2) Cabeza RoI: toma cada región propuesta, la clasifica y afina su caja.
Sobre un backbone ResNet-50 con FPN (pirámide de características) preentrenado en COCO.
Las cajas se usan en píxeles (xyxy) y la clase 0 es el "fondo": nuestras clases 0..27 pasan a 1..28.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from cnndet.data import augment_hsv   # mismo cambio de color que usamos en la CNN propia


from detkit.paths import resolve_dataset_cfg


def load_cfg(yaml_path):
    root, names, _ = resolve_dataset_cfg(yaml_path)
    return root, names


class FjordFRCNN(Dataset):
    def __init__(self, yaml_path, split, train=False):
        self.root, self.names = load_cfg(yaml_path)
        self.train = train
        self.files = sorted((self.root / "images" / split).glob("*.jpg"))
        self.labels = [self.root / "labels" / split / f"{f.stem}.txt" for f in self.files]
        assert self.files, f"No hay imágenes en {self.root / 'images' / split}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(str(self.files[i]))
        h, w = img.shape[:2]
        rows = [l.split() for l in self.labels[i].read_text().splitlines() if l.strip()]
        a = np.array(rows, np.float32).reshape(-1, 5)
        boxes = np.stack([(a[:, 1] - a[:, 3] / 2) * w, (a[:, 2] - a[:, 4] / 2) * h,
                          (a[:, 1] + a[:, 3] / 2) * w, (a[:, 2] + a[:, 4] / 2) * h], 1)
        cls = a[:, 0].astype(np.int64)
        meta = dict(name=self.files[i].stem, gt_boxes=boxes.copy(), gt_cls=cls.copy())
        if self.train:
            img = augment_hsv(img)
            if random.random() < 0.5:   # volteo horizontal
                img = img[:, ::-1]
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)   # sin cajas degeneradas
        boxes, cls = boxes[keep], cls[keep]
        rgb = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))                    # BGR->RGB, CHW, uint8
        target = dict(boxes=torch.from_numpy(boxes.astype(np.float32)).reshape(-1, 4),
                      labels=torch.from_numpy(cls + 1))                                  # +1: el 0 es "fondo"
        return torch.from_numpy(rgb), target, meta


def collate(batch):
    return tuple(zip(*batch))


def build_model(num_classes, pretrained=True, trainable_layers=3, img_size=640, anchors="small"):
    """ResNet-50 + FPN. Con pretrained=True parte de los pesos de COCO (se descargan solos la primera vez).

    trainable_layers: cuántos bloques del backbone se entrenan, contando desde el final
      (3 = layer4, layer3 y layer2 entrenables; conv1 y layer1 congeladas). Las BatchNorm quedan congeladas.
    anchors: 'small' usa anclas de 16 a 256 px (nuestros objetos son pequeños), 'default' usa 32 a 512.
    """
    sizes = ((16,), (32,), (64,), (128,), (256,)) if anchors == "small" else ((32,), (64,), (128,), (256,), (512,))
    kw = dict(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None, weights_backbone=None,
              min_size=img_size, max_size=img_size,           # lado mayor = img_size (igual que YOLO y la CNN)
              box_score_thresh=0.001, box_detections_per_img=100,   # mismo protocolo de evaluación que los demás
              rpn_anchor_generator=AnchorGenerator(sizes, ((0.5, 1.0, 2.0),) * 5))
    if pretrained:
        kw["trainable_backbone_layers"] = trainable_layers
    model = fasterrcnn_resnet50_fpn(**kw)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes + 1)   # capa NUEVA: 28 clases + fondo
    return model


def trainable_report(model):
    """Parámetros entrenables / totales por componente (evidencia de qué se congeló y qué se entrena)."""
    groups = {}
    for n, p in model.named_parameters():
        key = ".".join(n.split(".")[:3]) if n.startswith("backbone.body") else ".".join(n.split(".")[:2])
        t, tot = groups.get(key, (0, 0))
        groups[key] = (t + (p.numel() if p.requires_grad else 0), tot + p.numel())
    return {k: dict(trainable=v[0], total=v[1], state="ENTRENA" if v[0] else "CONGELADO") for k, v in groups.items()}


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    preds, gts, names = [], [], []
    for images, _, metas in loader:
        out = model([im.to(dev).float() / 255 for im in images])
        for o, m in zip(out, metas):
            preds.append(dict(boxes=o["boxes"].cpu().numpy(), scores=o["scores"].cpu().numpy(),
                              labels=(o["labels"] - 1).cpu().numpy()))
            gts.append(dict(boxes=m["gt_boxes"], labels=m["gt_cls"]))
            names.append(m["name"])
    return preds, gts, names


@torch.no_grad()
def val_loss(model, loader, dev):
    """torchvision solo devuelve pérdidas en modo train (la BatchNorm está congelada, así que es seguro)."""
    model.train()
    tot, n = 0.0, 0
    for images, targets, _ in loader:
        images = [im.to(dev).float() / 255 for im in images]
        targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
        tot += sum(model(images, targets).values()).item()
        n += 1
    return tot / n
