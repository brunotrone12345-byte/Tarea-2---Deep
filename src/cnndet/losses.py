"""losses.py - Pérdida estilo CenterNet y decodificación de predicciones."""
import torch
import torch.nn.functional as F

from .data import STRIDE


def focal_loss(logits, gt, eps=1e-4):
    """Focal loss con penalización reducida alrededor del centro (CenterNet).
    Hay MUCHÍSIMAS más celdas 'vacías' que centros; la focal loss evita que las vacías dominen."""
    p = torch.sigmoid(logits.float()).clamp(eps, 1 - eps)
    pos = gt.eq(1).float()
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * (1 - pos)
    n = pos.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / n.clamp(min=1)


def gather(feat, ind):
    """feat (B,C,H,W), ind (B,K) -> (B,K,C): valor de la salida en la celda de cada objeto."""
    f = feat.flatten(2).permute(0, 2, 1)
    return f.gather(1, ind.unsqueeze(2).expand(-1, -1, f.shape[2]))


def centernet_loss(out, batch, w_hm=1.0, w_wh=1.0, w_off=1.0):
    mask = batch["mask"].unsqueeze(2)
    n = mask.sum().clamp(min=1)
    l_hm = focal_loss(out["hm"], batch["hm"])
    pred_wh = gather(out["wh"].float(), batch["ind"])
    l_wh = (torch.abs(pred_wh - torch.log(batch["wh"].clamp(min=1e-3))) * mask).sum() / n
    pred_off = gather(out["off"].float(), batch["ind"])
    l_off = (torch.abs(pred_off - batch["off"]) * mask).sum() / n
    total = w_hm * l_hm + w_wh * l_wh + w_off * l_off
    return total, dict(hm=l_hm.item(), wh=l_wh.item(), off=l_off.item())


@torch.no_grad()
def decode(out, K=100):
    """Devuelve cajas xyxy (en píxeles del lienzo), scores y clases: (B,K,4), (B,K), (B,K)."""
    heat = torch.sigmoid(out["hm"].float())
    B, C, H, W = heat.shape
    peaks = (F.max_pool2d(heat, 3, 1, 1) == heat).float()      # solo máximos locales (sustituye al NMS)
    scores, idx = (heat * peaks).view(B, -1).topk(K)
    cls, pix = idx // (H * W), idx % (H * W)
    ys, xs = (pix // W).float(), (pix % W).float()
    wh = gather(out["wh"].float(), pix).clamp(max=5).exp()
    off = gather(out["off"].float(), pix)
    cx, cy = (xs + off[..., 0]) * STRIDE, (ys + off[..., 1]) * STRIDE
    w, h = wh[..., 0] * STRIDE, wh[..., 1] * STRIDE
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    return boxes, scores, cls


def to_original(boxes, meta):
    """Del lienzo a píxeles de la imagen original. boxes (K,4); meta = [sx,sy,x0,y0,w0,h0]."""
    sx, sy, x0, y0, w0, h0 = meta.tolist()
    b = boxes.clone()
    b[:, [0, 2]] = ((b[:, [0, 2]] - x0) / sx).clamp(0, w0)
    b[:, [1, 3]] = ((b[:, [1, 3]] - y0) / sy).clamp(0, h0)
    return b
