"""
eval_degradation.py - Etapa 7 (Ítems 3.1, 3.2 y 3.3 de la Rúbrica): evaluación de robustez ante degradación.

Evalúa el mejor modelo (`yolo26s_tl_full` por defecto, o cualquier otro detector del proyecto)
frente a 4 familias físicas de degradación submarina × 3 niveles de severidad (12 condiciones + baseline limpio)
sobre el conjunto de prueba (`test`, 451 imágenes), SIN modificar ni duplicar imágenes en `data/fjord/`.

Usa EXACTAMENTE el mismo evaluador común (`detkit/metrics.py`) y mantiene fijo el umbral de confianza
seleccionado en `val` (`results/eval/<name>_val_threshold.json`, `thr = 0.30` en YOLO26s transfer).

Salidas por defecto:
  - results/degradation/degradation_summary.csv  y  degradation_table.md
  - results/degradation/degradation_per_class.csv
  - figures/degradation/fig_degradation_samples.jpg
  - figures/degradation/fig_degradation_curves.png
  - figures/degradation/fig_degradation_drop_bars.png

Uso (desde la raíz del proyecto):
    python src/eval_degradation.py
    python src/eval_degradation.py --weights runs/yolo/yolo26s_tl_full/weights/best.pt --device 0
"""
import argparse
import json
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
from torchmetrics.detection import MeanAveragePrecision

from detkit.metrics import coco_map, pr_at_threshold
from detkit.paths import resolve_dataset_cfg

MIN_N_FREQ = 20
REF_SIZE = 640.0

# 4 tipos del enunciado × 3 niveles de severidad = 12 condiciones + baseline limpio
CONDITIONS = [
    dict(key="clean", family="Sin degradación", family_id="clean", level=0, level_name="Limpio",
         param_str="Original", desc="Conjunto test original sin alterar"),
    # 1. Reducción de iluminación (atenuación de luz con la profundidad)
    dict(key="low_light_1", family="Baja iluminación", family_id="low_light", level=1, level_name="Leve",
         param=1.5, param_str="gamma=1.5", desc="Corrección gamma 1.5"),
    dict(key="low_light_2", family="Baja iluminación", family_id="low_light", level=2, level_name="Moderado",
         param=2.2, param_str="gamma=2.2", desc="Corrección gamma 2.2"),
    dict(key="low_light_3", family="Baja iluminación", family_id="low_light", level=3, level_name="Severo",
         param=3.0, param_str="gamma=3.0", desc="Corrección gamma 3.0"),
    # 2. Desenfoque gaussiano (corrientes submarinas / fuera de foco)
    dict(key="blur_1", family="Desenfoque gaussiano", family_id="gaussian_blur", level=1, level_name="Leve",
         param=1.5, param_str="sigma=1.5 px", desc="Filtro gaussiano sigma=1.5 (k=11)"),
    dict(key="blur_2", family="Desenfoque gaussiano", family_id="gaussian_blur", level=2, level_name="Moderado",
         param=3.0, param_str="sigma=3.0 px", desc="Filtro gaussiano sigma=3.0 (k=19)"),
    dict(key="blur_3", family="Desenfoque gaussiano", family_id="gaussian_blur", level=3, level_name="Severo",
         param=5.0, param_str="sigma=5.0 px", desc="Filtro gaussiano sigma=5.0 (k=31)"),
    # 3. Ruido electrónico de sensor (ganancia ISO alta en oscuridad submarina)
    dict(key="noise_1", family="Ruido gaussiano", family_id="gaussian_noise", level=1, level_name="Leve",
         param=15.0, param_str="sigma=15", desc="Ruido aditivo N(0, 15^2)"),
    dict(key="noise_2", family="Ruido gaussiano", family_id="gaussian_noise", level=2, level_name="Moderado",
         param=30.0, param_str="sigma=30", desc="Ruido aditivo N(0, 30^2)"),
    dict(key="noise_3", family="Ruido gaussiano", family_id="gaussian_noise", level=3, level_name="Severo",
         param=50.0, param_str="sigma=50", desc="Ruido aditivo N(0, 50^2)"),
    # 4. Disminución de contraste / turbidez submarina (velo acuático por partículas en suspensión)
    dict(key="turbidity_1", family="Bajo contraste / turbidez", family_id="turbidity", level=1, level_name="Leve",
         param=0.35, param_str="alpha=0.35 (65% contr.)", desc="Velo submarino alpha=0.35"),
    dict(key="turbidity_2", family="Bajo contraste / turbidez", family_id="turbidity", level=2, level_name="Moderado",
         param=0.55, param_str="alpha=0.55 (45% contr.)", desc="Velo submarino alpha=0.55"),
    dict(key="turbidity_3", family="Bajo contraste / turbidez", family_id="turbidity", level=3, level_name="Severo",
         param=0.75, param_str="alpha=0.75 (25% contr.)", desc="Velo submarino alpha=0.75"),
]

FAMILY_COLORS = {
    "low_light": "#1f77b4",
    "gaussian_blur": "#d62728",
    "gaussian_noise": "#ff7f0e",
    "turbidity": "#2ca02c",
}


def apply_degradation(img_bgr, cond, seed_offset=0):
    """Aplica en memoria la degradación especificada sobre una imagen BGR uint8."""
    fid = cond["family_id"]
    if fid == "clean":
        return img_bgr
    p = cond["param"]
    if fid == "low_light":
        lut = np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** p) * 255.0, 0, 255).astype(np.uint8)
        return cv2.LUT(img_bgr, lut)
    if fid == "gaussian_blur":
        k = int(2 * np.ceil(3 * p) + 1)
        return cv2.GaussianBlur(img_bgr, (k, k), sigmaX=p, sigmaY=p)
    if fid == "gaussian_noise":
        rng = np.random.default_rng(seed_offset)
        noise = rng.normal(0.0, p, size=img_bgr.shape).astype(np.float32)
        return np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if fid == "turbidity":
        # Luz ambiental acuática (sesgo azul-verdoso característico del medio marino)
        mean_bgr = img_bgr.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        veil = np.clip(mean_bgr * np.array([[[1.08, 1.03, 0.88]]], dtype=np.float32), 0, 255)
        out = (1.0 - p) * img_bgr.astype(np.float32) + p * veil
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(f"Degradación desconocida: {fid}")


def read_gt_boxes(label_path, w, h):
    if not Path(label_path).exists():
        return dict(boxes=np.zeros((0, 4), np.float32), labels=np.zeros((0,), np.int64))
    rows = [l.split() for l in Path(label_path).read_text().splitlines() if l.strip()]
    if not rows:
        return dict(boxes=np.zeros((0, 4), np.float32), labels=np.zeros((0,), np.int64))
    a = np.array(rows, np.float32).reshape(-1, 5)
    boxes = np.stack([
        (a[:, 1] - a[:, 3] / 2) * w,
        (a[:, 2] - a[:, 4] / 2) * h,
        (a[:, 1] + a[:, 3] / 2) * w,
        (a[:, 2] + a[:, 4] / 2) * h,
    ], axis=1)
    return dict(boxes=boxes, labels=a[:, 0].astype(np.int64))


def build_predictor(kind, weights_path, imgsz, device, bs):
    """Devuelve una función `predict_images(list_of_bgr_arrays) -> list_of_pred_dicts`."""
    weights_path = Path(weights_path)
    assert weights_path.exists(), (
        f"No se encontró el archivo de pesos: {weights_path}\n"
        f"Ejecuta primero el cuaderno notebooks/Etapa7_Colab_YOLO26s_TL_y_Degradacion.ipynb "
        f"o copia best.pt en esa ruta."
    )
    if kind == "yolo":
        from ultralytics import YOLO
        model = YOLO(str(weights_path))

        def predict_fn(imgs):
            preds = []
            for i in range(0, len(imgs), bs):
                chunk = imgs[i:i + bs]
                res = model.predict(source=chunk, imgsz=imgsz, conf=0.001, max_det=300,
                                    verbose=False, device=device)
                for r in res:
                    preds.append(dict(
                        boxes=r.boxes.xyxy.cpu().numpy().astype(np.float32),
                        scores=r.boxes.conf.cpu().numpy().astype(np.float32),
                        labels=r.boxes.cls.cpu().numpy().astype(np.int64),
                    ))
            return preds

        return predict_fn

    if kind == "cnn":
        from cnndet.data import place
        from cnndet.losses import decode, to_original
        from cnndet.model import CenterDet
        dev = "cuda" if (device != "cpu" and torch.cuda.is_available()) else "cpu"
        ck = torch.load(weights_path, map_location=dev)
        model = CenterDet(ck["nc"], width=ck["width"]).to(dev).eval()
        model.load_state_dict(ck["model"])
        S = ck["img_size"]

        @torch.no_grad()
        def predict_fn(imgs):
            preds = []
            for img in imgs:
                h0, w0 = img.shape[:2]
                s = S / max(h0, w0)
                nw, nh = max(round(w0 * s), 1), max(round(h0 * s), 1)
                im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                x0, y0 = int(round(S / 2 - nw / 2)), int(round(S / 2 - nh / 2))
                canvas = place(im, S, x0, y0)
                x = torch.from_numpy(np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1))).to(dev).float().div(255)[None]
                out = model(x)
                boxes, scores, cls = decode(out, 100)
                keep = scores[0] > 0.001
                bb = to_original(boxes[0][keep].cpu(), torch.tensor([nw / w0, nh / h0, x0, y0, w0, h0]))
                ss, cc = scores[0][keep].cpu(), cls[0][keep].cpu()
                if len(bb):
                    k = batched_nms(bb, ss, cc, 0.6)
                    bb, ss, cc = bb[k], ss[k], cc[k]
                preds.append(dict(boxes=bb.numpy(), scores=ss.numpy(), labels=cc.numpy().astype(np.int64)))
            return preds

        return predict_fn

    if kind == "frcnn":
        from frcnn.lib import build_model as build_frcnn
        dev = "cuda" if (device != "cpu" and torch.cuda.is_available()) else "cpu"
        ck = torch.load(weights_path, map_location=dev)
        model = build_frcnn(ck["num_classes"], ck["pretrained"], 3, ck["img_size"], ck["anchors"]).to(dev).eval()
        model.load_state_dict(ck["model"])

        @torch.no_grad()
        def predict_fn(imgs):
            preds = []
            for img in imgs:
                x = torch.from_numpy(np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))).to(dev).float().div(255)
                o = model([x])[0]
                preds.append(dict(
                    boxes=o["boxes"].cpu().numpy(),
                    scores=o["scores"].cpu().numpy(),
                    labels=(o["labels"] - 1).cpu().numpy().astype(np.int64),
                ))
            return preds

        return predict_fn

    raise ValueError(f"Tipo de modelo no soportado: {kind}")


def compute_small_map50(preds, gts, sizes_wh):
    """Calcula mAP@0.5 para objetos pequeños (<32 px a escala 640) y medianos (32-96 px)."""
    sp, sg = [], []
    for p, g, (w, h) in zip(preds, gts, sizes_wh):
        s = REF_SIZE / max(w, h)
        sp.append(dict(
            boxes=torch.as_tensor(p["boxes"] * s, dtype=torch.float32).reshape(-1, 4),
            scores=torch.as_tensor(p["scores"], dtype=torch.float32).reshape(-1),
            labels=torch.as_tensor(p["labels"], dtype=torch.int64).reshape(-1),
        ))
        sg.append(dict(
            boxes=torch.as_tensor(g["boxes"] * s, dtype=torch.float32).reshape(-1, 4),
            labels=torch.as_tensor(g["labels"], dtype=torch.int64).reshape(-1),
        ))
    m50 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5])
    m50.warn_on_many_detections = False
    m50.update(sp, sg)
    r = m50.compute()
    return float(r["map_small"]), float(r["map_medium"]), float(r["map_large"])


def draw_tile(img_bgr, gt, pred, class_names, thr, title, w_tile=420):
    k = w_tile / img_bgr.shape[1]
    im = cv2.resize(img_bgr, (w_tile, round(img_bgr.shape[0] * k)))
    for b in gt["boxes"]:
        x1, y1, x2, y2 = (b * k).astype(int)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 210, 0), 2)
    for b, s, l in zip(pred["boxes"], pred["scores"], pred["labels"]):
        if s < thr:
            continue
        x1, y1, x2, y2 = (b * k).astype(int)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(im, f"{class_names[int(l)]} {s:.2f}", (x1, max(y1 - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    banner = np.zeros((28, w_tile, 3), dtype=np.uint8)
    cv2.putText(banner, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, im])


def save_visual_mosaic(sample_img, sample_gt, sample_preds, class_names, thr, out_path):
    """Genera un mosaico 4 filas (familias) × 4 columnas (Limpio, Leve, Moderado, Severo)."""
    families = [
        ("low_light", "Baja luz"),
        ("gaussian_blur", "Desenfoque"),
        ("gaussian_noise", "Ruido"),
        ("turbidity", "Turbidez"),
    ]
    cond_by_fl = {(c["family_id"], c["level"]): c for c in CONDITIONS}
    clean_tile = draw_tile(sample_img, sample_gt, sample_preds["clean"], class_names, thr, "Original (sin degradar)")

    grid_rows = []
    for fid, short_name in families:
        row_tiles = [clean_tile.copy()]
        for lv in (1, 2, 3):
            c = cond_by_fl[(fid, lv)]
            deg_im = apply_degradation(sample_img, c, seed_offset=42000 + lv)
            ttl = f"{short_name} - {c['level_name']} ({c['param_str']})"
            row_tiles.append(draw_tile(deg_im, sample_gt, sample_preds[c["key"]], class_names, thr, ttl))
        grid_rows.append(np.hstack(row_tiles))
    cv2.imwrite(str(out_path), np.vstack(grid_rows))


def plot_degradation_figures(df, out_fig):
    clean_row = df[df["key"] == "clean"].iloc[0]

    # Figura 1: Curvas de caída por nivel de severidad (0=Limpio, 1=Leve, 2=Moderado, 3=Severo)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True)
    metrics = [
        ("map50", "mAP@0.5 en test"),
        ("map50_95", "mAP@0.5:0.95 en test"),
        ("f1", "F1 en test (umbral fijo val)"),
    ]
    families = [
        ("low_light", "Baja iluminación"),
        ("gaussian_blur", "Desenfoque gaussiano"),
        ("gaussian_noise", "Ruido gaussiano"),
        ("turbidity", "Bajo contraste / turbidez"),
    ]
    for ax, (m_col, ylabel) in zip(axs, metrics):
        for fid, flabel in families:
            sub = df[df["family_id"] == fid].sort_values("level")
            xs = [0] + sub["level"].tolist()
            ys = [clean_row[m_col]] + sub[m_col].tolist()
            ax.plot(xs, ys, "o-", lw=2, color=FAMILY_COLORS[fid], label=flabel)
        ax.axhline(clean_row[m_col], ls="--", lw=1, color="#555555", alpha=0.6)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["Limpio", "Leve", "Moderado", "Severo"])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_ylim(0, max(float(df[m_col].max()) * 1.15, 0.45))
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle("Robustez del detector frente a condiciones crecientes de degradación submarina", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_fig / "fig_degradation_curves.png", dpi=150)
    plt.close()

    # Figura 2: Ranking de caída porcentual por condición respecto al baseline limpio
    deg = df[df["key"] != "clean"].copy()
    deg = deg.sort_values("drop_pct_map50_95", ascending=True)  # más negativo arriba al invertir eje
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    y = np.arange(len(deg))
    labels = [f"{r['family']} ({r['level_name']})" for _, r in deg.iterrows()]
    bar_colors = [FAMILY_COLORS[r["family_id"]] for _, r in deg.iterrows()]

    for ax, col, ttl in zip(
        axs,
        ["drop_pct_map50", "drop_pct_map50_95"],
        ["Variación relativa en mAP@0.5 (%)", "Variación relativa en mAP@0.5:0.95 (%)"],
    ):
        bars = ax.barh(y, deg[col], color=bar_colors, alpha=0.88)
        ax.axvline(0, color="black", lw=0.8)
        for b_rect, v in zip(bars, deg[col]):
            offset = -1.2 if v < 0 else 1.2
            ha = "right" if v < 0 else "left"
            ax.text(v + offset, b_rect.get_y() + b_rect.get_height() / 2, f"{v:+.1f}%",
                    va="center", ha=ha, fontsize=8)
        ax.set_xlabel(ttl)
        ax.set_title(ttl)
        ax.set_xlim(min(deg[col].min() - 14, -20), max(deg[col].max() + 8, 5))
        ax.grid(axis="x", alpha=0.3)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(labels, fontsize=8.5)
    axs[0].invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_fig / "fig_degradation_drop_bars.png", dpi=150)
    plt.close()


def df_to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="yolo", choices=["yolo", "cnn", "frcnn"])
    ap.add_argument("--model-name", default="yolo26s_tl_full", help="Nombre para leer el umbral val JSON")
    ap.add_argument("--weights", default="runs/yolo/yolo26s_tl_full/weights/best.pt")
    ap.add_argument("--yaml", default="data/fjord/data.yaml")
    ap.add_argument("--thr", type=float, default=None, help="Umbral fijo de val (si se omite, lee *_val_threshold.json)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="0 = todas las 451 imágenes de test; >0 = prueba rápida")
    ap.add_argument("--out-dir", default=".", help="Raíz donde escribir results/degradation y figures/degradation")
    args = ap.parse_args()

    out_base = Path(args.out_dir)
    out_res = out_base / "results" / "degradation"
    out_fig = out_base / "figures" / "degradation"
    out_res.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    if args.thr is None:
        thr_file = Path("results/eval") / f"{args.model_name}_val_threshold.json"
        args.thr = float(json.load(open(thr_file))["thr"]) if thr_file.exists() else 0.30

    root, names, _ = resolve_dataset_cfg(args.yaml)
    files = sorted((root / "images" / "test").glob("*.jpg"))
    if args.limit > 0:
        files = files[:args.limit]
    assert files, f"No hay imágenes en {root / 'images' / 'test'}"

    print(f"Cargando {len(files)} imágenes de test en RAM...")
    orig_imgs, gts, sizes_wh = [], [], []
    for f in files:
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        orig_imgs.append(im)
        gts.append(read_gt_boxes(root / "labels" / "test" / f"{f.stem}.txt", w, h))
        sizes_wh.append((w, h))

    # Conteos GT por clase para filtrar clases frecuentes (n >= 20)
    gt_counts = np.zeros(len(names), dtype=int)
    for g in gts:
        for c in g["labels"]:
            gt_counts[int(c)] += 1
    freq_classes = {names[c] for c in range(len(names)) if gt_counts[c] >= MIN_N_FREQ}

    # Elegir imagen representativa con varios peces (3..8 objetos) para el mosaico visual
    n_objs = [len(g["labels"]) for g in gts]
    cand_idx = [i for i, k in enumerate(n_objs) if 3 <= k <= 8]
    sample_idx = cand_idx[len(cand_idx) // 2] if cand_idx else int(np.argmax(n_objs))

    predict_fn = build_predictor(args.kind, args.weights, args.imgsz, args.device, args.bs)
    print(f"Evaluando {len(CONDITIONS)} condiciones con umbral val fijo = {args.thr:.2f} | pesos = {args.weights}")

    summary_rows = []
    per_class_rows = []
    sample_preds = {}
    clean_metrics = None

    for cond in CONDITIONS:
        t0 = time.time()
        deg_imgs = [
            apply_degradation(im, cond, seed_offset=args.seed * 10000 + idx)
            for idx, im in enumerate(orig_imgs)
        ]
        preds = predict_fn(deg_imgs)
        sample_preds[cond["key"]] = preds[sample_idx]

        m = coco_map(preds, gts, names)
        pr = pr_at_threshold(preds, gts, args.thr, len(names))
        map50_s, map50_m, map50_l = compute_small_map50(preds, gts, sizes_wh)

        freq_ap50 = [v["ap50"] for cname, v in m["per_class"].items() if cname in freq_classes]
        freq_ap = [v["ap50_95"] for cname, v in m["per_class"].items() if cname in freq_classes]
        map50_freq = float(np.mean(freq_ap50)) if freq_ap50 else np.nan
        map50_95_freq = float(np.mean(freq_ap)) if freq_ap else np.nan

        if cond["key"] == "clean":
            clean_metrics = dict(map50=m["map50"], map50_95=m["map"], f1=pr["f1"])

        drop_50 = 100.0 * (m["map50"] - clean_metrics["map50"]) / max(clean_metrics["map50"], 1e-9)
        drop_50_95 = 100.0 * (m["map"] - clean_metrics["map50_95"]) / max(clean_metrics["map50_95"], 1e-9)
        drop_f1 = 100.0 * (pr["f1"] - clean_metrics["f1"]) / max(clean_metrics["f1"], 1e-9)
        dt = time.time() - t0

        row = dict(
            key=cond["key"],
            family_id=cond["family_id"],
            family=cond["family"],
            level=cond["level"],
            level_name=cond["level_name"],
            param_str=cond["param_str"],
            map50=m["map50"],
            map50_95=m["map"],
            map75=m["map75"],
            precision=pr["precision"],
            recall=pr["recall"],
            f1=pr["f1"],
            map50_freq20=map50_freq,
            map50_95_freq20=map50_95_freq,
            map50_small=map50_s,
            map50_medium=map50_m,
            map50_large=map50_l,
            drop_pct_map50=0.0 if cond["key"] == "clean" else drop_50,
            drop_pct_map50_95=0.0 if cond["key"] == "clean" else drop_50_95,
            drop_pct_f1=0.0 if cond["key"] == "clean" else drop_f1,
            eval_time_s=dt,
        )
        summary_rows.append(row)

        for cid, cname in enumerate(names):
            if gt_counts[cid] == 0:
                continue
            ap_c = m["per_class"].get(cname, {})
            per_class_rows.append(dict(
                condition=cond["key"],
                family=cond["family"],
                level=cond["level"],
                clase=cname,
                n_gt=int(gt_counts[cid]),
                ap50=ap_c.get("ap50", np.nan),
                ap50_95=ap_c.get("ap50_95", np.nan),
            ))

        print(
            f"  [{cond['key']:<13}] mAP@0.5={row['map50']:.3f} ({row['drop_pct_map50']:+5.1f}%) | "
            f"mAP@0.5:0.95={row['map50_95']:.3f} ({row['drop_pct_map50_95']:+5.1f}%) | "
            f"F1={row['f1']:.3f} | {dt:.1f}s",
            flush=True,
        )

    df = pd.DataFrame(summary_rows)
    df.round(4).to_csv(out_res / "degradation_summary.csv", index=False)
    pd.DataFrame(per_class_rows).round(4).to_csv(out_res / "degradation_per_class.csv", index=False)

    disp = pd.DataFrame({
        "Condición": df["family"],
        "Nivel": df["level_name"],
        "Parámetro": df["param_str"],
        "mAP@0.5": df["map50"].map(lambda v: f"{v:.3f}"),
        "Δ mAP@0.5": df["drop_pct_map50"].map(lambda v: "—" if abs(v) < 1e-9 else f"{v:+.1f}%"),
        "mAP@0.5:0.95": df["map50_95"].map(lambda v: f"{v:.3f}"),
        "Δ mAP@0.5:0.95": df["drop_pct_map50_95"].map(lambda v: "—" if abs(v) < 1e-9 else f"{v:+.1f}%"),
        "Precisión": df["precision"].map(lambda v: f"{v:.3f}"),
        "Recall": df["recall"].map(lambda v: f"{v:.3f}"),
        "F1": df["f1"].map(lambda v: f"{v:.3f}"),
        "mAP@0.5 (n>=20)": df["map50_freq20"].map(lambda v: f"{v:.3f}"),
        "mAP@0.5 (<32px)": df["map50_small"].map(lambda v: f"{v:.3f}"),
    })
    md_text = (
        f"# Resultados de Robustez ante Degradación en Test ({len(files)} imágenes, umbral val = {args.thr:.2f})\n\n"
        f"- **Modelo evaluado:** `{args.model_name}` (`{args.weights}`)\n\n"
        + df_to_markdown(disp)
    )
    (out_res / "degradation_table.md").write_text(md_text, encoding="utf-8")

    save_visual_mosaic(orig_imgs[sample_idx], gts[sample_idx], sample_preds, names, max(args.thr, 0.30),
                       out_fig / "fig_degradation_samples.jpg")
    plot_degradation_figures(df, out_fig)
    print(f"\nListo: tablas en {out_res} y figuras en {out_fig}")


if __name__ == "__main__":
    main()
