"""
analyze_results.py - Etapa 6, paso 6.3 (sin GPU): recall por tamaño de objeto e intervalos de confianza.

Usa las predicciones guardadas (results/preds/*_test.csv), las etiquetas de test y las particiones.

  1) Efecto del tope de detecciones por imagen (YOLO emite hasta 300; la CNN y Faster R-CNN, 100).
  2) Recall por tamaño de objeto (¿fallan más los peces pequeños? ¿es un problema de detectar o de clasificar?).
  3) Intervalos de confianza al 95% del mAP por BOOTSTRAP DE BLOQUES: se remuestrean los bloques de imágenes
     consecutivas (las imágenes de un mismo bloque son casi iguales; remuestrear imágenes sueltas subestimaría
     la incertidumbre). Las diferencias entre modelos se calculan con los MISMOS remuestreos (pareadas).

Uso (desde la carpeta del proyecto):
    python src/analyze_results.py                # 500 remuestreos
    python src/analyze_results.py --boot 1000
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from detkit.metrics import coco_map

ROOT = Path(".")
MODELS = [("CNN propia", "cnn_v1"), ("YOLO26s desde cero", "yolo26s_scratch"),
          ("YOLO26s transfer", "yolo26s_tl_full"), ("Faster R-CNN transfer", "frcnn_v1")]
REF = "YOLO26s transfer"
IOUS = np.arange(0.5, 0.96, 0.05)
BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]
BIN_NAMES = ["<16", "16-32", "32-64", "64-128", ">=128"]


# ---------------------------------------------------------------- datos
def load_test():
    sp = pd.read_csv(ROOT / "data/splits/splits_resized.csv", dtype={"stem": str})
    sp = sp[sp.split == "test"].reset_index(drop=True)
    gt = {}
    for r in sp.itertuples():
        rows = [l.split() for l in (ROOT / "data/fjord/labels/test" / f"{r.stem}.txt").read_text().splitlines() if l.strip()]
        a = np.array(rows, np.float32).reshape(-1, 5)
        W, H = r.new_W, r.new_H
        boxes = np.stack([(a[:, 1] - a[:, 3] / 2) * W, (a[:, 2] - a[:, 4] / 2) * H,
                          (a[:, 1] + a[:, 3] / 2) * W, (a[:, 2] + a[:, 4] / 2) * H], 1)
        gt[r.stem] = (boxes, a[:, 0].astype(int), W, H)
    return list(sp.stem), sp.group.values, gt


def load_preds(name, stems, max_det):
    df = pd.read_csv(ROOT / "results/preds" / f"{name}_test.csv", dtype={"image": str})
    out = {s: (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)) for s in stems}
    for img, g in df.groupby("image"):
        if img not in out:
            continue
        g = g.sort_values("score", ascending=False).head(max_det)
        out[img] = (g[["x1", "y1", "x2", "y2"]].values.astype(float), g.score.values.astype(float), g.class_id.values.astype(int))
    return out


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa, ab = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


# ---------------------------------------------------------------- mAP rápido (estilo COCO) con emparejamiento previo
def build_records(stems, gt, preds, nc):
    """Empareja detecciones y objetos reales UNA vez por imagen, clase y umbral de IoU (como COCO)."""
    img_i, cls_i, score, tp = [], [], [], []
    ngt = np.zeros((len(stems), nc))
    for k, st in enumerate(stems):
        gb, gc = gt[st][0], gt[st][1]
        for c in gc:
            ngt[k, c] += 1
        pb, ps, pc = preds[st]
        for c in np.unique(pc):
            m = pc == c
            db, ds, g = pb[m], ps[m], gb[gc == c]          # ya vienen ordenadas por score descendente
            flags = np.zeros((len(db), len(IOUS)), bool)
            if len(g):
                I = iou(db, g)
                for t, thr in enumerate(IOUS):
                    used = np.zeros(len(g), bool)
                    for d in range(len(db)):
                        cand = np.where(used, -1.0, I[d])
                        j = int(cand.argmax())
                        if cand[j] >= thr - 1e-10:
                            used[j] = True
                            flags[d, t] = True
            img_i += [k] * len(db)
            cls_i += [c] * len(db)
            score += list(ds)
            tp.append(flags)
    rec = dict(img=np.array(img_i, int), cls=np.array(cls_i, int), score=np.array(score), ngt=ngt,
               tp=np.vstack(tp) if tp else np.zeros((0, len(IOUS)), bool))
    # ordena por score dentro de cada clase una sola vez
    rec["by_class"] = {}
    for c in range(nc):
        idx = np.flatnonzero(rec["cls"] == c)
        idx = idx[np.argsort(-rec["score"][idx], kind="stable")]
        rec["by_class"][c] = (rec["img"][idx], rec["tp"][idx])
    return rec


def ap_class(img, tp, mult, n_gt):
    """AP (101 puntos) para cada umbral de IoU; mult = veces que aparece cada imagen en el remuestreo."""
    w = mult[img].astype(float)
    keep = w > 0
    if not keep.any():
        return np.zeros(len(IOUS))
    tp, w = tp[keep], w[keep]
    tpc, fpc = np.cumsum(tp * w[:, None], 0), np.cumsum((~tp) * w[:, None], 0)
    rec, prec = tpc / n_gt, tpc / np.maximum(tpc + fpc, 1e-12)
    prec = np.maximum.accumulate(prec[::-1], 0)[::-1]
    rs = np.linspace(0, 1, 101)
    ap = np.zeros(len(IOUS))
    for t in range(len(IOUS)):
        idx = np.searchsorted(rec[:, t], rs, side="left")
        ap[t] = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1), t], 0.0).mean()
    return ap


def map_scores(rec, mult):
    n_gt = rec["ngt"].T @ mult                      # objetos reales por clase en el remuestreo
    aps = [ap_class(*rec["by_class"][c], mult, n_gt[c]) if len(rec["by_class"][c][0]) else np.zeros(len(IOUS))
           for c in range(len(n_gt)) if n_gt[c] > 0]
    a = np.array(aps)
    return a[:, 0].mean(), a.mean()


# ---------------------------------------------------------------- recall por tamaño
def recall_by_size(stems, gt, preds, thr):
    sizes, hit_any, hit_cls = [], [], []
    for st in stems:
        gb, gc, W, H = gt[st]
        pb, ps, pc = preds[st]
        keep = ps >= thr
        pb, pc = pb[keep], pc[keep]
        for b, c in zip(gb, gc):
            sizes.append(np.sqrt((b[2] - b[0]) * (b[3] - b[1])) * 640 / max(W, H))
            if len(pb):
                I = iou(b[None], pb)[0]
                hit_any.append(I.max() >= 0.5)
                hit_cls.append((I[pc == c].max() >= 0.5) if (pc == c).any() else False)
            else:
                hit_any.append(False)
                hit_cls.append(False)
    return np.array(sizes), np.array(hit_any), np.array(hit_cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--validate", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_t, out_f = ROOT / "results/comparison", ROOT / "figures/comparison"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    stems, group, gt = load_test()
    cfg = yaml.safe_load(open(ROOT / "data/fjord/data.yaml", encoding="utf-8"))
    names = [cfg["names"][k] for k in sorted(cfg["names"])] if isinstance(cfg["names"], dict) else cfg["names"]
    nc = len(names)
    print(f"Test: {len(stems)} imágenes, {len(set(group))} bloques, {sum(len(v[1]) for v in gt.values())} objetos")

    # ---------- 1) efecto del tope de detecciones + validación del estimador
    recs, rows = {}, []
    ones = np.ones(len(stems))
    for label, name in MODELS:
        r100 = build_records(stems, gt, load_preds(name, stems, args.max_det), nc)
        r300 = build_records(stems, gt, load_preds(name, stems, 300), nc)
        recs[label] = r100
        m100, m300 = map_scores(r100, ones), map_scores(r300, ones)
        row = dict(modelo=label, **{f"mAP@0.5 (tope {args.max_det})": m100[0], f"mAP@0.5:0.95 (tope {args.max_det})": m100[1],
                                    "mAP@0.5 (tope 300)": m300[0], "mAP@0.5:0.95 (tope 300)": m300[1]})
        if args.validate:
            P = []
            pr = load_preds(name, stems, args.max_det)
            for s in stems:
                P.append(dict(boxes=pr[s][0], scores=pr[s][1], labels=pr[s][2]))
            G = [dict(boxes=gt[s][0], labels=gt[s][1]) for s in stems]
            tm = coco_map(P, G, names)
            row["torchmetrics mAP@0.5"], row["torchmetrics mAP@0.5:0.95"] = tm["map50"], tm["map"]
        rows.append(row)
    cap = pd.DataFrame(rows)
    cap.round(4).to_csv(out_t / "detection_cap_check.csv", index=False)
    print("\n=== 1) Tope de detecciones por imagen y validación del estimador rápido ===")
    print(cap.round(3).to_string(index=False))

    # ---------- 2) recall por tamaño
    rows = []
    for label, name in MODELS:
        thr = json.load(open(ROOT / "results/eval" / f"{name}_val_threshold.json"))["thr"]
        s, ha, hc = recall_by_size(stems, gt, load_preds(name, stems, args.max_det), thr)
        for (lo, hi), bn in zip(BINS, BIN_NAMES):
            m = (s >= lo) & (s < hi)
            rows.append(dict(modelo=label, tamano_px=bn, n_objetos=int(m.sum()),
                             recall_deteccion=ha[m].mean() if m.any() else np.nan,
                             recall_clase_correcta=hc[m].mean() if m.any() else np.nan, umbral=thr))
    rs = pd.DataFrame(rows)
    rs.round(3).to_csv(out_t / "recall_by_size.csv", index=False)
    print("\n=== 2) Recall (IoU>=0.5) por tamaño del objeto (px, imagen a 640) ===")
    print(rs.pivot(index="tamano_px", columns="modelo", values="recall_clase_correcta").loc[BIN_NAMES].round(2).to_string())
    print("(arriba: con la clase correcta; el recall solo de detección está en el CSV y en la figura)")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cols = dict(zip([m[0] for m in MODELS], ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]))
    nb = rs[rs.modelo == MODELS[0][0]].set_index("tamano_px").loc[BIN_NAMES].n_objetos
    for ax, c, ttl in zip(axs, ["recall_deteccion", "recall_clase_correcta"],
                          ["Recall solo de detección (cualquier clase)", "Recall con la clase correcta"]):
        for label, _ in MODELS:
            d = rs[rs.modelo == label].set_index("tamano_px").loc[BIN_NAMES]
            ax.plot(BIN_NAMES, d[c], "o-", label=label, color=cols[label])
        ax.set_xticks(range(len(BIN_NAMES)))
        ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(BIN_NAMES, nb)])
        ax.set_xlabel("tamaño del objeto (px, imagen a 640)")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=.3)
    axs[0].set_ylabel("recall (IoU >= 0.5)")
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_f / "fig_recall_by_size.png", dpi=150)
    plt.close()

    # ---------- 3) bootstrap de bloques
    rng = np.random.default_rng(args.seed)
    ug, inv = np.unique(group, return_inverse=True)
    res = {m[0]: [] for m in MODELS}
    for b in range(args.boot):
        gm = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))   # veces que sale cada bloque
        mult = gm[inv].astype(float)
        for label, _ in MODELS:
            res[label].append(map_scores(recs[label], mult))
        if (b + 1) % 100 == 0:
            print(f"  remuestreo {b + 1}/{args.boot}")
    res = {k: np.array(v) for k, v in res.items()}
    rows, drows = [], []
    for label, _ in MODELS:
        est = map_scores(recs[label], ones)
        lo, hi = np.percentile(res[label], [2.5, 97.5], axis=0)
        rows.append(dict(modelo=label, **{"mAP@0.5": est[0], "IC95 inf": lo[0], "IC95 sup": hi[0],
                                           "mAP@0.5:0.95": est[1], "IC95 inf ": lo[1], "IC95 sup ": hi[1]}))
        if label != REF:
            d = res[REF] - res[label]
            dl, dh = np.percentile(d, [2.5, 97.5], axis=0)
            drows.append(dict(comparacion=f"{REF} - {label}",
                              **{"dif mAP@0.5": map_scores(recs[REF], ones)[0] - est[0], "IC95 inf": dl[0], "IC95 sup": dh[0],
                                 "P(YOLO transfer mejor) mAP@0.5": (d[:, 0] > 0).mean(),
                                 "dif mAP@0.5:0.95": map_scores(recs[REF], ones)[1] - est[1], "IC95 inf ": dl[1], "IC95 sup ": dh[1],
                                 "P(YOLO transfer mejor) mAP@0.5:0.95": (d[:, 1] > 0).mean()}))
    bs, bd = pd.DataFrame(rows), pd.DataFrame(drows)
    bs.round(4).to_csv(out_t / "bootstrap_summary.csv", index=False)
    bd.round(4).to_csv(out_t / "bootstrap_diffs.csv", index=False)
    print(f"\n=== 3) Intervalos de confianza al 95% (bootstrap de bloques, {args.boot} remuestreos) ===")
    print(bs.round(3).to_string(index=False))
    print("\n=== Diferencias pareadas (positivo = YOLO transfer mejor) ===")
    print(bd.round(3).to_string(index=False))

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (c, lo, hi, ttl) in zip(axs, [("mAP@0.5", "IC95 inf", "IC95 sup", "mAP@0.5"),
                                          ("mAP@0.5:0.95", "IC95 inf ", "IC95 sup ", "mAP@0.5:0.95")]):
        for i, r in bs.iterrows():
            ax.errorbar(r[c], i, xerr=[[r[c] - r[lo]], [r[hi] - r[c]]], fmt="o", color=cols[r["modelo"]], capsize=4)
        ax.set_yticks(range(len(bs)))
        ax.set_yticklabels(bs.modelo)
        ax.invert_yaxis()
        ax.set_xlabel(f"{ttl} en test (IC 95%)")
        ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_f / "fig_bootstrap_ci.png", dpi=150)
    plt.close()
    print(f"\nListo: tablas en {out_t}, figuras en {out_f}")


if __name__ == "__main__":
    main()
