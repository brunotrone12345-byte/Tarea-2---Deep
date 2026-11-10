"""
compare_models.py - Etapa 6, paso 6.1: tabla maestra y gráficos comparativos de los cuatro modelos.

Usa SOLO archivos ya generados (results/eval/*, runs/*/summary.json y las curvas de entrenamiento).
No necesita GPU. Uso (desde la carpeta del proyecto):
    python src/compare_models.py

Salidas:
    results/comparison/master_table.csv  y  master_table.md
    figures/comparison/fig_map_bars.png, fig_time_vs_map.png, fig_per_class_ap50.png, fig_curves.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(".")
# hist: formato del archivo de curvas ('own' = nuestros scripts, 'ultra' = results.csv de Ultralytics)
MODELS = [
    dict(label="CNN propia", kind="desde cero", name="cnn_v1", run="runs/cnn/cnn_v1", hist="history.csv", fmt="own"),
    dict(label="YOLO26s desde cero", kind="desde cero", name="yolo26s_scratch", run="runs/yolo/yolo26s_scratch",
         hist="results.csv", fmt="ultra"),
    dict(label="YOLO26s transfer", kind="transfer", name="yolo26s_tl_full", run="runs/yolo/yolo26s_tl_full",
         hist="results.csv", fmt="ultra"),
    dict(label="Faster R-CNN transfer", kind="transfer", name="frcnn_v1", run="runs/frcnn/frcnn_v1",
         hist="history.csv", fmt="own"),
]
ABLATION = dict(label="YOLO26s transfer (backbone congelado)", name="yolo26s_tl_frozen",
                run="runs/yolo/yolo26s_tl_frozen")
COLORS = dict(zip([m["label"] for m in MODELS], ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]))
MIN_N = 20   # umbral de objetos para el promedio "solo clases frecuentes"


def jload(p):
    return json.load(open(p, encoding="utf-8")) if Path(p).exists() else None


def ev(name, split):
    return jload(ROOT / "results" / "eval" / f"{name}_{split}_metrics.json")


def main():
    out_t, out_f = ROOT / "results" / "comparison", ROOT / "figures" / "comparison"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    rows, per_class = [], {}
    for m in MODELS:
        s, val, test = jload(ROOT / m["run"] / "summary.json"), ev(m["name"], "val"), ev(m["name"], "test")
        assert s and val and test, f"Faltan archivos de {m['name']} (summary.json o evaluación en val/test)"
        pc = pd.read_csv(ROOT / "results" / "eval" / f"{m['name']}_test_per_class.csv")
        per_class[m["label"]] = pc
        fr = pc[pc.n_gt >= MIN_N]
        rows.append({
            "modelo": m["label"], "tipo": m["kind"],
            "mAP@0.5 test": test["map50"], "mAP@0.5:0.95 test": test["map50_95"], "mAP@0.75 test": test["map75"],
            "precision test": test["precision"], "recall test": test["recall"], "F1 test": test["f1"],
            f"mAP@0.5 test (clases con n>={MIN_N})": fr.ap50.mean(),
            f"mAP@0.5:0.95 test (clases con n>={MIN_N})": fr.ap50_95.mean(),
            "mAP@0.5 val": val["map50"], "mAP@0.5:0.95 val": val["map50_95"],
            "fitness val": 0.1 * val["map50"] + 0.9 * val["map50_95"],
            "params (M)": s["params"] / 1e6, "epocas": s.get("epochs_run", s.get("epochs")),
            "mejor epoca": s["best_epoch"], "tiempo total (min)": s["total_time_min"],
            "ms/imagen (provisional)": test["inference_ms_per_image"],
        })
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out_t / "master_table.csv", index=False)

    # ---- Tabla en Markdown (para pegar en la presentación) ----
    cols = [c for c in df.columns if c not in ("tipo",)]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(r[c] if isinstance(r[c], str) else
                                    (f"{r[c]:.0f}" if c in ("epocas", "mejor epoca") else f"{r[c]:.3f}" if abs(r[c]) < 10 else f"{r[c]:.1f}")
                                    for c in cols) + " |")
    ab = {sp: ev(ABLATION["name"], sp) for sp in ("val",)}
    sab = jload(ROOT / ABLATION["run"] / "summary.json")
    if ab["val"] and sab:
        a = ab["val"]
        md += ["", f"Ablación (solo val): {ABLATION['label']}: mAP@0.5 = {a['map50']:.3f}, mAP@0.5:0.95 = {a['map50_95']:.3f}, "
                   f"fitness = {0.1 * a['map50'] + 0.9 * a['map50_95']:.3f}, tiempo total = {sab['total_time_min']:.1f} min."]
    (out_t / "master_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---- Figura 1: mAP por modelo (test) ----
    metrics = ["mAP@0.5 test", "mAP@0.5:0.95 test", "mAP@0.75 test"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.2
    for i, r in df.iterrows():
        vals = [r[c] for c in metrics]
        b = ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, label=r["modelo"], color=COLORS[r["modelo"]])
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["mAP@0.5", "mAP@0.5:0.95", "mAP@0.75"])
    ax.set_ylabel("mAP en test")
    ax.set_title("Desempeño de los cuatro modelos en el conjunto de prueba")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_f / "fig_map_bars.png", dpi=150)
    plt.close()

    # ---- Figura 2: tiempo de entrenamiento vs mAP (tamaño = parámetros) ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, c in zip(axs, ["mAP@0.5 test", "mAP@0.5:0.95 test"]):
        for _, r in df.iterrows():
            ax.scatter(r["tiempo total (min)"], r[c], s=60 + r["params (M)"] * 14, color=COLORS[r["modelo"]], alpha=.8)
            ax.annotate(r["modelo"], (r["tiempo total (min)"], r[c]), textcoords="offset points", xytext=(8, 6), fontsize=8)
        ax.margins(x=0.3, y=0.15)
        ax.set_xlabel("tiempo total de entrenamiento (min, con validación)")
        ax.set_ylabel(c)
        ax.grid(alpha=.3)
    fig.suptitle("Costo de entrenamiento vs desempeño (el tamaño del círculo es proporcional a los parámetros)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_f / "fig_time_vs_map.png", dpi=150)
    plt.close()

    # ---- Figura 3: AP@0.5 por clase (test) ----
    base = per_class[MODELS[0]["label"]][["clase", "n_gt"]].sort_values("n_gt", ascending=False)
    mat = np.array([[per_class[m["label"]].set_index("clase").ap50.get(c, np.nan) for m in MODELS] for c in base.clase])
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(base) + 1.8))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] < 0.55 else "black")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m["label"] for m in MODELS], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(base)))
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(base.clase, base.n_gt)], fontsize=8)
    ax.set_title("AP@0.5 por clase en test (n = objetos)")
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(out_f / "fig_per_class_ap50.png", dpi=150)
    plt.close()

    # ---- Figura 4: mAP@0.5:0.95 en val según el tiempo de entrenamiento ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in MODELS:
        p = ROOT / m["run"] / m["hist"]
        if not p.exists():
            continue
        h = pd.read_csv(p)
        h.columns = [c.strip() for c in h.columns]
        if m["fmt"] == "ultra":
            x, y = h["time"] / 60, h["metrics/mAP50-95(B)"]
        else:
            h = h.dropna(subset=["val_map"])
            x, y = h["elapsed_min"], h["val_map"]
        ax.plot(x, y, label=m["label"], color=COLORS[m["label"]])
    ax.set_xlabel("tiempo de entrenamiento (min)")
    ax.set_ylabel("mAP@0.5:0.95 en val")
    ax.set_title("Velocidad de convergencia")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.text(0.01, 0.005, "Cada modelo usa el evaluador de su propia biblioteca (Ultralytics usa el suyo): solo para comparar tendencias.",
             fontsize=6.5)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(out_f / "fig_curves.png", dpi=150)
    plt.close()
    print(f"\nListo: {out_t / 'master_table.md'} y 4 figuras en {out_f}")


if __name__ == "__main__":
    main()
