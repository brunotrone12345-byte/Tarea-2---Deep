"""
train_cnn.py - Paso 2.2: entrena la CNN propia DESDE CERO, valida y registra métricas y tiempos.

Uso (desde la carpeta del proyecto):
    python src/train_cnn.py --epochs 3 --name smoke      # prueba rápida
    python src/train_cnn.py --epochs 100 --name cnn_v1   # entrenamiento real

Usa SOLO train y val. El conjunto de test no se toca aquí.
Resultados en runs/cnn/<name>/: best.pt, last.pt, history.csv, summary.json, curves.png
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cnndet.data import FjordDataset, collate
from cnndet.losses import centernet_loss
from cnndet.model import CenterDet
from cnndet.predict import predict_loader
from detkit.metrics import coco_map


class EMA:
    """Media móvil exponencial de los pesos: suaviza el ruido del entrenamiento y suele dar mejor mAP."""
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay, self.updates = decay, 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 500))   # al inicio sigue más rápido al modelo
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def to_dev(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}


@torch.no_grad()
def val_loss(model, loader, dev, amp):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        bt = to_dev(batch, dev)
        with torch.autocast(**amp):
            out = model(bt["img"].float() / 255)
        tot += centernet_loss(out, bt)[0].item()
        n += 1
    return tot / n


def plot_curves(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoch, h.train_loss, label="train")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_xlabel("época"); ax[0].set_ylabel("pérdida"); ax[0].set_yscale("log"); ax[0].legend()
    m = h.dropna(subset=["val_map"])
    ax[1].plot(m.epoch, m.val_map50, "o-", label="mAP@0.5")
    ax[1].plot(m.epoch, m.val_map, "o-", label="mAP@0.5:0.95")
    ax[1].set_xlabel("época"); ax[1].set_ylabel("mAP (val)"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="data/fjord/data.yaml")
    ap.add_argument("--name", default="cnn_v1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=float, default=2, help="épocas de calentamiento del lr")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5, help="cada cuántas épocas calcular mAP en val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    run = Path(args.out) / "runs" / "cnn" / args.name
    run.mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "figures").mkdir(exist_ok=True)

    ds_tr = FjordDataset(args.yaml, "train", train=True, img_size=args.img_size)
    ds_va = FjordDataset(args.yaml, "val", train=False, img_size=args.img_size)
    kw = dict(collate_fn=collate, num_workers=args.workers, persistent_workers=args.workers > 0)
    tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=True, pin_memory=dev == "cuda", **kw)
    va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, **kw)

    model = CenterDet(ds_tr.nc, width=args.width).to(dev)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    total_it, warm_it = args.epochs * len(tr), int(args.warmup * len(tr))
    json.dump(dict(vars(args), params=n_params, n_train=len(ds_tr), n_val=len(ds_va)),
              open(run / "config.json", "w"), indent=2)
    print(f"Entrenando '{args.name}' en {dev}: {n_params / 1e6:.2f} M parámetros | "
          f"{len(ds_tr)} train / {len(ds_va)} val | {len(tr)} iteraciones por época")

    rows, best_fit, best_ep, train_time, it = [], -1.0, 0, 0.0, 0
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, sums = time.time(), np.zeros(4)
        for batch in tr:
            lr = (args.lr * (it + 1) / warm_it if it < warm_it else
                  args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (it - warm_it) / max(total_it - warm_it, 1)))))
            for g in opt.param_groups:
                g["lr"] = lr
            bt = to_dev(batch, dev)
            with torch.autocast(**amp):
                out = model(bt["img"].float() / 255)
            loss, parts = centernet_loss(out, bt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            ema.update(model)
            sums += [loss.item(), parts["hm"], parts["wh"], parts["off"]]
            it += 1
        if dev == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - t0
        train_time += ep_time
        sums /= len(tr)

        row = dict(epoch=ep, lr=lr, train_loss=sums[0], train_hm=sums[1], train_wh=sums[2], train_off=sums[3],
                   val_loss=val_loss(ema.module, va, dev, amp), val_map50=np.nan, val_map=np.nan,
                   fitness=np.nan, epoch_time_s=ep_time)
        if ep % args.eval_every == 0 or ep == args.epochs:
            preds, gts, _ = predict_loader(ema.module, va, dev)
            r = coco_map(preds, gts, ds_va.names)
            row.update(val_map50=r["map50"], val_map=r["map"], fitness=0.1 * r["map50"] + 0.9 * r["map"])
            if row["fitness"] > best_fit:
                best_fit, best_ep = row["fitness"], ep
                torch.save(dict(model=ema.module.state_dict(), nc=ds_tr.nc, width=args.width,
                                img_size=args.img_size, epoch=ep, names=ds_tr.names), run / "best.pt")
        row["elapsed_min"] = (time.time() - t_start) / 60
        rows.append(row)
        h = pd.DataFrame(rows)
        h.to_csv(run / "history.csv", index=False)
        torch.save(dict(model=ema.module.state_dict(), nc=ds_tr.nc, width=args.width,
                        img_size=args.img_size, epoch=ep, names=ds_tr.names), run / "last.pt")
        msg = f"ep {ep:3d}/{args.epochs}  loss {row['train_loss']:.3f}  val {row['val_loss']:.3f}  {ep_time:4.0f}s"
        if not np.isnan(row["val_map"]):
            msg += f"  | val mAP50 {row['val_map50']:.3f}  mAP50-95 {row['val_map']:.3f}  (mejor: ep {best_ep})"
        print(msg, flush=True)

    total = time.time() - t_start
    summary = dict(name=args.name, params=n_params, epochs=args.epochs, best_epoch=best_ep, best_fitness=best_fit,
                   train_time_min=train_time / 60, total_time_min=total / 60,
                   mean_epoch_s=train_time / args.epochs,
                   peak_vram_gb=(torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None))
    json.dump(summary, open(run / "summary.json", "w"), indent=2)
    plot_curves(pd.DataFrame(rows), run / "curves.png")
    plot_curves(pd.DataFrame(rows), Path(args.out) / "figures" / f"{args.name}_curves.png")
    print("\n=== RESUMEN ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
