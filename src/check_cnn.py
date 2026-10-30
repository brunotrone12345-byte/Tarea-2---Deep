"""
check_cnn.py - Paso 2.1: verifica datos, modelo, pérdida y hace una prueba de sobreajuste.

Uso (desde la carpeta del proyecto):
    python src/check_cnn.py
Opciones útiles:  --img-size 640  --width 1.0  --bs 16  --iters 200  --workers 0
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

from cnndet.data import FjordDataset, collate
from cnndet.losses import centernet_loss, decode
from cnndet.model import CenterDet


def draw_check(ds, out_path, n=4):
    """Imagen aumentada con sus cajas (izquierda) y el mapa de calor objetivo (derecha)."""
    rows = []
    for i in np.linspace(0, len(ds) - 1, n).astype(int):
        s = ds[i]
        img = np.ascontiguousarray(s["img"].numpy().transpose(1, 2, 0)[..., ::-1])  # RGB -> BGR
        vis = img.copy()
        for b, c in zip(s["canvas_boxes"].numpy(), s["canvas_cls"].numpy()):
            x1, y1, x2, y2 = b.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, ds.names[c], (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        heat = cv2.resize((s["hm"].max(0).values.numpy() * 255).astype(np.uint8), (ds.S, ds.S),
                          interpolation=cv2.INTER_NEAREST)
        over = cv2.addWeighted(img, 0.5, cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.5, 0)
        rows.append(np.hstack([vis, over]))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_path), cv2.resize(sheet, None, fx=1400 / sheet.shape[1], fy=1400 / sheet.shape[1]))


def to_dev(batch, dev):
    return {k: v.to(dev) for k, v in batch.items() if torch.is_tensor(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="data/fjord/data.yaml")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    fig_dir = Path(args.out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    print("== A. DATOS ==")
    ds = FjordDataset(args.yaml, "train", train=True, img_size=args.img_size)
    print(f"{len(ds)} imágenes de train | {ds.nc} clases | salida de la red: {ds.out}x{ds.out}")
    t0 = time.time()
    for i in range(20):
        ds[i]
    print(f"Carga + aumento: {(time.time() - t0) / 20 * 1000:.0f} ms por imagen")
    draw_check(ds, fig_dir / "cnn_aug_check.jpg")
    print("Guardado figures/cnn_aug_check.jpg  (REVISA que las cajas sigan sobre los peces)")

    print("\n== B. MODELO ==")
    model = CenterDet(ds.nc, width=args.width).to(dev)
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M | dispositivo: {dev}")
    batch = collate([ds[i] for i in range(min(args.bs, len(ds)))])
    bt = to_dev(batch, dev)
    with torch.no_grad(), torch.autocast(**amp):
        out = model(bt["img"].float() / 255)
    for k, v in out.items():
        print(f"  salida {k}: {tuple(v.shape)}")

    print("\n== C. PÉRDIDA, MEMORIA Y VELOCIDAD ==")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for it in range(12):
        if it == 2 and dev == "cuda":
            torch.cuda.synchronize()
        if it == 2:
            t0 = time.time()
        with torch.autocast(**amp):
            out = model(bt["img"].float() / 255)
        loss, parts = centernet_loss(out, bt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it == 0:
            print(f"Pérdida inicial: {loss.item():.3f}  {parts}")
    if dev == "cuda":
        torch.cuda.synchronize()
        print(f"Memoria GPU pico: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB (batch {len(batch['name'])})")
    print(f"Velocidad: {(time.time() - t0) / 10:.2f} s por iteración (solo cómputo, sin cargar datos)")

    print(f"\n== D. PRUEBA DE SOBREAJUSTE ({args.iters} iteraciones sobre 16 imágenes fijas) ==")
    ds_fix = FjordDataset(args.yaml, "train", train=False, img_size=args.img_size)
    batch = collate([ds_fix[i] for i in np.linspace(0, len(ds_fix) - 1, 16).astype(int)])
    bt = to_dev(batch, dev)
    model = CenterDet(ds.nc, width=args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    for it in range(1, args.iters + 1):
        with torch.autocast(**amp):
            out = model(bt["img"].float() / 255)
        loss, parts = centernet_loss(out, bt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if it == 1 or it % 25 == 0:
            print(f"  it {it:4d}  pérdida {loss.item():7.3f}  hm {parts['hm']:.3f}  wh {parts['wh']:.3f}  off {parts['off']:.3f}")

    model.eval()
    with torch.no_grad(), torch.autocast(**amp):
        out = model(bt["img"].float() / 255)
    boxes, scores, cls = decode(out, K=100)
    hit = n_gt = n_pred = n_ok = 0
    for b in range(len(batch["name"])):
        keep = scores[b] > 0.3
        pb, pc = boxes[b][keep], cls[b][keep]
        gb, gc = batch["canvas_boxes"][b].to(dev), batch["canvas_cls"][b].to(dev)
        n_gt += len(gb)
        n_pred += len(pb)
        if len(pb) and len(gb):
            iou = box_iou(gb, pb) * (gc[:, None] == pc[None, :])
            hit += int((iou.max(1).values >= 0.5).sum())
            n_ok += int((iou.max(0).values >= 0.5).sum())
    print(f"\nRecall (IoU>=0.5, clase correcta): {hit}/{n_gt} = {hit / max(n_gt, 1):.0%}")
    print(f"Predicciones con score>0.3: {n_pred} | correctas: {n_ok} ({n_ok / max(n_pred, 1):.0%})")
    print("Esperado: la pérdida baja mucho y el recall sube claramente. Si no, hay un error en datos/modelo/pérdida.")


if __name__ == "__main__":
    main()
