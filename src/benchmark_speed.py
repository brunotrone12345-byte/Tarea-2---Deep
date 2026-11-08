"""
benchmark_speed.py - Etapa 6, paso 6.2: velocidad de inferencia con UN SOLO protocolo para los 4 modelos.

Protocolo: batch 1, precisión fp32, imágenes reales del conjunto de test ya cargadas en RAM (BGR, numpy).
Se cronometra TODO el camino: preprocesado -> GPU -> red -> decodificación/NMS -> cajas finales en CPU (numpy).
Así ningún modelo se beneficia de que parte de su trabajo quede fuera de la medición.

Uso (desde la carpeta del proyecto; cierra otros programas que usen la GPU):
    python src/benchmark_speed.py                      # GPU, 150 imágenes
    python src/benchmark_speed.py --device cpu --n 30  # opcional: velocidad en CPU
"""
import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms

from cnndet.data import place
from cnndet.losses import decode, to_original
from cnndet.model import CenterDet
from frcnn.lib import build_model as build_frcnn

ROOT = Path(".")
SPECS = [
    dict(label="CNN propia", kind="cnn", path="runs/cnn/cnn_v1/best.pt"),
    dict(label="YOLO26s desde cero", kind="yolo", path="runs/yolo/yolo26s_scratch/weights/best.pt"),
    dict(label="YOLO26s transfer", kind="yolo", path="runs/yolo/yolo26s_tl_full/weights/best.pt"),
    dict(label="Faster R-CNN transfer", kind="frcnn", path="runs/frcnn/frcnn_v1/best.pt"),
]


def make_cnn(path, dev, amp):
    ck = torch.load(ROOT / path, map_location=dev)
    model = CenterDet(ck["nc"], width=ck["width"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    S = ck["img_size"]

    @torch.no_grad()
    def infer(img):
        h0, w0 = img.shape[:2]
        s = S / max(h0, w0)
        nw, nh = max(round(w0 * s), 1), max(round(h0 * s), 1)
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        x0, y0 = int(round(S / 2 - nw / 2)), int(round(S / 2 - nh / 2))
        canvas = place(im, S, x0, y0)
        x = torch.from_numpy(np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1))).to(dev).float().div(255)[None]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and dev == "cuda"):
            out = model(x)
        boxes, scores, cls = decode(out, 100)
        keep = scores[0] > 0.001
        bb = to_original(boxes[0][keep], torch.tensor([nw / w0, nh / h0, x0, y0, w0, h0]))
        ss, cc = scores[0][keep], cls[0][keep]
        if len(bb):
            k = batched_nms(bb, ss, cc, 0.6)
            bb, ss, cc = bb[k], ss[k], cc[k]
        return bb.cpu().numpy(), ss.cpu().numpy(), cc.cpu().numpy()

    return model, infer


def make_yolo(path, dev, imgsz):
    from ultralytics import YOLO
    m = YOLO(ROOT / path)

    def infer(img):
        r = m.predict(source=img, imgsz=imgsz, conf=0.001, max_det=300, verbose=False,
                      device=0 if dev == "cuda" else "cpu")[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()

    return m.model, infer


def make_frcnn(path, dev):
    ck = torch.load(ROOT / path, map_location=dev)
    model = build_frcnn(ck["num_classes"], ck["pretrained"], 3, ck["img_size"], ck["anchors"]).to(dev).eval()
    model.load_state_dict(ck["model"])

    @torch.no_grad()
    def infer(img):
        x = torch.from_numpy(np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))).to(dev).float().div(255)
        o = model([x])[0]
        return o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy()

    return model, infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--n", type=int, default=150, help="imágenes cronometradas")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640, help="tamaño de entrada de YOLO")
    ap.add_argument("--amp-cnn", type=int, default=0, help="1 = bf16 en la CNN (en la evaluación se usó bf16)")
    ap.add_argument("--data", default="data/fjord")
    args = ap.parse_args()
    dev = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("Midiendo en CPU.")

    files = sorted((ROOT / args.data / "images" / "test").glob("*.jpg"))
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(args.n + args.warmup, len(files))).astype(int)]
    imgs = [cv2.imread(str(f)) for f in files]
    print(f"{len(imgs)} imágenes de test en RAM ({args.warmup} de calentamiento)")

    rows = []
    for sp in SPECS:
        print(f"\n== {sp['label']} ==")
        if sp["kind"] == "cnn":
            model, infer = make_cnn(sp["path"], dev, bool(args.amp_cnn))
        elif sp["kind"] == "yolo":
            model, infer = make_yolo(sp["path"], dev, args.imgsz)
        else:
            model, infer = make_frcnn(sp["path"], dev)
        n_params = sum(p.numel() for p in model.parameters())
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for i, im in enumerate(imgs):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            infer(im)
            if dev == "cuda":
                torch.cuda.synchronize()
            if i >= args.warmup:
                ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        rows.append(dict(modelo=sp["label"], ms_media=ts.mean(), ms_mediana=np.median(ts), ms_p95=np.percentile(ts, 95),
                         ms_std=ts.std(), fps=1000 / ts.mean(), params_M=n_params / 1e6,
                         mem_pico_MB=torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else np.nan, n=len(ts)))
        r = rows[-1]
        print(f"  {r['ms_media']:.1f} ms de media (mediana {r['ms_mediana']:.1f}, p95 {r['ms_p95']:.1f}) -> {r['fps']:.0f} FPS")
        del model, infer
        if dev == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "gpu" if dev == "cuda" else "cpu"
    df.round(2).to_csv(out_dir / f"speed_benchmark_{tag}.csv", index=False)
    info = dict(device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor(),
                torch=torch.__version__, precision="fp32" + (" (CNN en bf16)" if args.amp_cnn else ""),
                batch=1, images=int(df.n.iloc[0]), protocol="imagen BGR en RAM -> cajas finales en CPU, con NMS")
    json.dump(info, open(out_dir / f"speed_benchmark_{tag}.json", "w"), indent=2)
    print("\n" + df.round(2).to_string(index=False))

    master = out_dir / "master_table.csv"
    if master.exists():
        m = pd.read_csv(master)[["modelo", "mAP@0.5 test", "mAP@0.5:0.95 test"]].merge(df, on="modelo")
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
        colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
        for ax, c in zip(axs, ["mAP@0.5 test", "mAP@0.5:0.95 test"]):
            for (_, r), col in zip(m.iterrows(), colors):
                ax.scatter(r["ms_media"], r[c], s=60 + r["params_M"] * 14, color=col, alpha=.8)
                ax.annotate(r["modelo"], (r["ms_media"], r[c]), textcoords="offset points", xytext=(8, 6), fontsize=8)
            ax.set_xlabel(f"tiempo de inferencia por imagen (ms, {tag.upper()}, batch 1)")
            ax.set_ylabel(c)
            ax.margins(x=0.3, y=0.15)
            ax.grid(alpha=.3)
        fig.suptitle("Precisión vs velocidad de inferencia (círculo proporcional a los parámetros)", fontsize=10)
        plt.tight_layout()
        (ROOT / "figures" / "comparison").mkdir(parents=True, exist_ok=True)
        plt.savefig(ROOT / "figures" / "comparison" / f"fig_speed_{tag}.png", dpi=150)
        plt.close()
    print(f"\nListo: results/comparison/speed_benchmark_{tag}.csv y figures/comparison/fig_speed_{tag}.png")


if __name__ == "__main__":
    main()
