"""predict.py - Inferencia de la CNN propia sobre un DataLoader (cajas en píxeles de la imagen original)."""
import torch
from torchvision.ops import batched_nms

from .losses import decode, to_original


@torch.no_grad()
def predict_loader(model, loader, dev, K=100, score_thr=0.01, nms_iou=0.6):
    """Devuelve (preds, gts, nombres). NMS por clase opcional (nms_iou=0 lo desactiva)."""
    model.eval()
    preds, gts, names = [], [], []
    for batch in loader:
        x = batch["img"].to(dev, non_blocking=True).float() / 255
        with torch.autocast(device_type="cuda", enabled=False):
            out = model(x)
        boxes, scores, cls = decode(out, K)
        for b in range(len(boxes)):
            keep = scores[b] > score_thr
            bb, ss, cc = to_original(boxes[b][keep], batch["meta"][b]), scores[b][keep], cls[b][keep]
            if nms_iou and len(bb):
                k = batched_nms(bb, ss, cc, nms_iou)
                bb, ss, cc = bb[k], ss[k], cc[k]
            preds.append(dict(boxes=bb.cpu().numpy(), scores=ss.cpu().numpy(), labels=cc.cpu().numpy()))
            gts.append(dict(boxes=batch["gt_boxes"][b].numpy(), labels=batch["gt_cls"][b].numpy()))
        names += batch["name"]
    return preds, gts, names
