"""
train_frcnn.py - Etapa 5: entrena Faster R-CNN (ResNet-50 FPN) con transfer learning desde COCO.

Uso (desde la carpeta del proyecto):
    python src/train_frcnn.py --epochs 1 --max-iters 40 --eval-every 1 --limit-val 40 --name smoke   # prueba
    python src/train_frcnn.py --epochs 20 --name frcnn_v1                                            # real

Usa solo train y val. El test no se toca aquí. Resultados en runs/frcnn/<name>/
"""
import argparse
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
from torch.utils.data import DataLoader, Subset

from detkit.metrics import coco_map
from frcnn.lib import FjordFRCNN, build_model, collate, predict, trainable_report, val_loss


def plot_curves(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoch, h.train_loss, label="train")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_xlabel("época"); ax[0].set_ylabel("pérdida"); ax[0].legend()
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
    ap.add_argument("--name", default="frcnn_v1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-iters", type=int, default=300)
    ap.add_argument("--trainable-layers", type=int, default=3)
    ap.add_argument("--anchors", default="small", choices=["small", "default"])
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=0, help="solo para pruebas: iteraciones por época")
    ap.add_argument("--limit-val", type=int, default=0, help="solo para pruebas: nº de imágenes de val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(args.amp) and dev == "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = Path(args.out) / "runs" / "frcnn" / args.name
    run.mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "figures").mkdir(exist_ok=True)

    ds_tr = FjordFRCNN(args.yaml, "train", train=True)
    ds_va = FjordFRCNN(args.yaml, "val", train=False)
    if args.limit_val:
        ds_va = Subset(ds_va, range(min(args.limit_val, len(ds_va))))
    names = ds_tr.names
    kw = dict(collate_fn=collate, num_workers=args.workers, persistent_workers=args.workers > 0)
    tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=True, pin_memory=dev == "cuda", **kw)
    va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, **kw)
    n_iter = args.max_iters or len(tr)

    model = build_model(len(names), bool(args.pretrained), args.trainable_layers, args.img_size, args.anchors).to(dev)
    rep = trainable_report(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(v["trainable"] for v in rep.values())
    print(f"Faster R-CNN ResNet-50 FPN | {n_params / 1e6:.1f} M parámetros, {n_train / 1e6:.1f} M entrenables | {dev}")
    for k, v in rep.items():
        print(f"  {k:<32} {v['state']:<10} {v['trainable'] / 1e6:6.2f} M de {v['total'] / 1e6:6.2f} M")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    total_it = args.epochs * n_iter
    json.dump(dict(vars(args), params=n_params, trainable_params=n_train, components=rep, n_train=len(ds_tr)),
              open(run / "config.json", "w"), indent=2)

    rows, best_fit, best_ep, train_time, it = [], -1.0, 0, 0.0, 0
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, sums, n = time.time(), np.zeros(5), 0
        for images, targets, _ in tr:
            if n >= n_iter:
                break
            lr = (args.lr * (0.001 + 0.999 * it / args.warmup_iters) if it < args.warmup_iters else
                  args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (it - args.warmup_iters) /
                                                                max(total_it - args.warmup_iters, 1)))))
            for g in opt.param_groups:
                g["lr"] = lr
            images = [im.to(dev, non_blocking=True).float() / 255 for im in images]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sums += [loss.item(), losses["loss_classifier"].item(), losses["loss_box_reg"].item(),
                     losses["loss_objectness"].item(), losses["loss_rpn_box_reg"].item()]
            it += 1
            n += 1
        if dev == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - t0
        train_time += ep_time
        sums /= max(n, 1)

        row = dict(epoch=ep, lr=lr, train_loss=sums[0], train_cls=sums[1], train_box=sums[2], train_obj=sums[3],
                   train_rpn_box=sums[4], val_loss=val_loss(model, va, dev), val_map50=np.nan, val_map=np.nan,
                   fitness=np.nan, epoch_time_s=ep_time)
        if ep % args.eval_every == 0 or ep == args.epochs:
            preds, gts, _ = predict(model, va, dev)
            r = coco_map(preds, gts, names)
            row.update(val_map50=r["map50"], val_map=r["map"], fitness=0.1 * r["map50"] + 0.9 * r["map"])
            if row["fitness"] > best_fit:
                best_fit, best_ep = row["fitness"], ep
                torch.save(dict(model=model.state_dict(), num_classes=len(names), img_size=args.img_size,
                                anchors=args.anchors, pretrained=bool(args.pretrained), epoch=ep, names=names),
                           run / "best.pt")
        row["elapsed_min"] = (time.time() - t_start) / 60
        rows.append(row)
        pd.DataFrame(rows).to_csv(run / "history.csv", index=False)
        torch.save(dict(model=model.state_dict(), num_classes=len(names), img_size=args.img_size,
                        anchors=args.anchors, pretrained=bool(args.pretrained), epoch=ep, names=names),
                   run / "last.pt")
        msg = f"ep {ep:3d}/{args.epochs}  loss {row['train_loss']:.3f}  val {row['val_loss']:.3f}  {ep_time:4.0f}s"
        if not np.isnan(row["val_map"]):
            msg += f"  | val mAP50 {row['val_map50']:.3f}  mAP50-95 {row['val_map']:.3f}  (mejor: ep {best_ep})"
        print(msg, flush=True)

    summary = dict(name=args.name, params=n_params, trainable_params=n_train, epochs=args.epochs,
                   best_epoch=best_ep, best_fitness=best_fit, train_time_min=train_time / 60,
                   total_time_min=(time.time() - t_start) / 60, mean_epoch_s=train_time / args.epochs,
                   peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None)
    json.dump(summary, open(run / "summary.json", "w"), indent=2)
    plot_curves(pd.DataFrame(rows), run / "curves.png")
    plot_curves(pd.DataFrame(rows), Path(args.out) / "figures" / f"{args.name}_curves.png")
    print("\n=== RESUMEN ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
