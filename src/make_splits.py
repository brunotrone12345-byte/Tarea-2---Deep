"""
make_splits.py - Paso 1.6: crea las particiones train/val/test (70/15/15).

Qué hace:
  1. Indexa imágenes y etiquetas por NOMBRE (stem), sin importar en qué carpeta oficial
     estén. Así la etiqueta de la imagen 2515, que Zenodo dejó en labels/val, se asocia
     correctamente a su imagen (que está en images/train).
  2. Agrupa las imágenes en bloques de IDs consecutivos (se corta si cambia la resolución).
     Un bloque se asigna ENTERO a un solo split: así fotogramas vecinos no quedan
     repartidos entre train y test.
  3. Opción A: las clases con menos de --min-ann anotaciones van solo a train y se
     declaran "no evaluables".
  4. Reparte los bloques con recocido simulado, buscando que cada clase evaluable quede
     repartida ~70/15/15 y que exista en val y en test.
  5. Audita la fuga de datos con las miniaturas de build_groups.py.

Uso:
    python src/make_splits.py --root "RUTA/A/FjordFish"
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

IMG_EXT = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "val", "test"]
TARGET = np.array([0.70, 0.15, 0.15])
# Penalización por clase evaluable AUSENTE en train / val / test (train y test son lo crítico)
PRES = np.array([3.0, 0.3, 1.0])


def index_files(root, kind):
    """{stem: ruta} recorriendo images/* o labels/*; exige nombres únicos."""
    found = {}
    for sub in sorted(d for d in (root / kind).iterdir() if d.is_dir()):
        for p in sub.iterdir():
            ok = p.suffix.lower() in IMG_EXT if kind == "images" else p.suffix.lower() == ".txt"
            if not ok:
                continue
            assert p.stem not in found, f"Nombre repetido: {p.stem}"
            found[p.stem] = p
    return found


def load_annotations(root, images, labels, nc):
    rows = []
    for stem in images:
        if stem not in labels:
            print(f"  [aviso] imagen sin etiqueta: {stem}")
            continue
        for line in labels[stem].read_text().splitlines():
            p = line.split()
            if not p:
                continue
            c = int(float(p[0]))
            assert 0 <= c < nc and len(p) == 5, f"Línea inválida en {stem}: {line}"
            rows.append((stem, c, *map(float, p[1:])))
    return pd.DataFrame(rows, columns=["stem", "cls", "xc", "yc", "w", "h"])


def chunk_groups(W, H, size):
    """Bloques de IDs consecutivos; nuevo bloque si cambia la resolución o llega a `size`."""
    g, cur, n = np.zeros(len(W), dtype=int), 0, 1
    for i in range(1, len(W)):
        if (W[i], H[i]) != (W[i - 1], H[i - 1]) or n >= size:
            cur += 1
            n = 0
        g[i] = cur
        n += 1
    return g


def anneal(M, forced, eval_mask, sizes, seed, iters=60000):
    """Asigna cada grupo a 0=train, 1=val, 2=test minimizando el desbalance por clase."""
    rng = np.random.default_rng(seed)
    G, nc = M.shape
    assign = np.where(forced, 0, rng.choice(3, size=G, p=TARGET))
    S, n = np.zeros((3, nc)), np.zeros(3)
    for s in range(3):
        S[s], n[s] = M[assign == s].sum(0), sizes[assign == s].sum()

    def loss():
        tot = S[:, eval_mask].sum(0)
        frac = S[:, eval_mask] / tot
        l = ((frac - TARGET[:, None]) ** 2).sum()          # proporción por clase
        l += 5 * ((n / n.sum() - TARGET) ** 2).sum()       # proporción de imágenes
        l += (PRES[:, None] * (S[:, eval_mask] == 0)).sum()  # clase ausente en un split
        return l

    free = np.flatnonzero(~forced)
    cur = loss()
    best, best_assign = cur, assign.copy()
    for it in range(iters):
        T = 0.02 * (1 - it / iters) + 1e-5
        g = rng.choice(free)
        old, new = assign[g], (assign[g] + rng.integers(1, 3)) % 3
        S[old] -= M[g]; S[new] += M[g]; n[old] -= sizes[g]; n[new] += sizes[g]
        nl = loss()
        if nl <= cur or rng.random() < np.exp((cur - nl) / T):
            assign[g], cur = new, nl
            if cur < best:
                best, best_assign = cur, assign.copy()
        else:
            S[old] += M[g]; S[new] -= M[g]; n[old] += sizes[g]; n[new] -= sizes[g]
    return best_assign, best


def nn_dist(T, held, train):
    """Distancia a la imagen de train más parecida, para cada imagen retenida."""
    return np.array([np.abs(T[train] - T[i]).mean(1).min() for i in held])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--chunk", type=int, default=25, help="imágenes consecutivas por bloque")
    ap.add_argument("--min-ann", type=int, default=10, help="clases con menos -> solo train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    for d in ("data/splits", "results", "figures"):
        (out / d).mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(root / "data.yaml", encoding="utf-8"))
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    nc = len(names)

    images, labels = index_files(root, "images"), index_files(root, "labels")
    print(f"Imágenes: {len(images)} | Etiquetas: {len(labels)}")
    print("Etiquetas sin imagen:", sorted(set(labels) - set(images)))
    ann = load_annotations(root, images, labels, nc)
    ann.to_csv(out / "results" / "annotations_all.csv", index=False)
    print(f"Cajas totales: {len(ann)}")

    meta = pd.read_csv(out / "results" / "dataset_images.csv", dtype={"stem": str})
    meta["id"] = meta.stem.astype(int)
    meta = meta.sort_values("id").reset_index(drop=True)   # mismo orden que thumbs.npy
    T = np.load(out / "results" / "thumbs.npy")
    assert len(T) == len(meta) == len(images), "Ejecuta de nuevo explore_dataset.py y build_groups.py"

    meta["group"] = chunk_groups(meta.W.values, meta.H.values, args.chunk)
    G = meta.group.max() + 1
    cls_idx = meta.reset_index().set_index("stem")["index"]
    ann["row"] = ann.stem.map(cls_idx)
    ann["group"] = meta.group.values[ann.row.values]
    M = np.zeros((G, nc))
    np.add.at(M, (ann.group.values, ann.cls.values), 1)
    sizes = meta.groupby("group").size().values

    total = M.sum(0)
    rare = (total > 0) & (total < args.min_ann)
    evalable = total >= args.min_ann
    forced = (M[:, rare].sum(1) > 0) if rare.any() else np.zeros(G, bool)
    print(f"\n{G} bloques de ~{args.chunk} imágenes. Clases con 1-{args.min_ann - 1} anotaciones "
          f"(solo train, no evaluables): {[f'{i}:{names[i]}({int(total[i])})' for i in np.flatnonzero(rare)]}")
    print(f"Bloques forzados a train por contener clases raras: {int(forced.sum())}")

    best = None
    for k in range(5):
        a, l = anneal(M, forced, evalable, sizes, args.seed + k)
        if best is None or l < best[1]:
            best = (a, l)
    assign = best[0]
    meta["split"] = [SPLITS[i] for i in assign[meta.group.values]]
    ann["split"] = meta.split.values[ann.row.values]

    # ---- Resumen ----
    print("\n=== PARTICIONES ===")
    print(meta.groupby("split").size().reindex(SPLITS).to_frame("imágenes").join(
        ann.groupby("split").size().reindex(SPLITS).rename("cajas")).to_string())
    tab = (ann.groupby(["cls", "split"]).size().unstack(fill_value=0)
           .reindex(index=range(nc), columns=SPLITS, fill_value=0))
    tab.insert(0, "clase", names)
    tab["total"] = tab[SPLITS].sum(axis=1)
    tab["estado"] = np.where(rare, "solo train (no evaluable)",
                             np.where(total == 0, "sin anotaciones", "ok"))
    print("\n=== CAJAS POR CLASE Y PARTICIÓN ===")
    print(tab.to_string())
    for sp in SPLITS:
        miss = tab[(tab.estado == "ok") & (tab[sp] == 0)]
        print(f"Clases evaluables ausentes en {sp}:",
              "ninguna" if miss.empty else miss.clase.tolist())
    tab.to_csv(out / "results" / "split_class_table.csv")

    # ---- Auditoría de fuga de datos ----
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(meta))
    official = pd.Series({s: p.parent.name for s, p in images.items()})[meta.stem].values
    perm = rng.permutation(idx)
    n_tr, n_va = int(0.70 * len(idx)), int(0.15 * len(idx))
    rand_train, rand_test = perm[:n_tr], perm[n_tr + n_va:]
    sets = {
        "oficial (val vs train)": (np.flatnonzero(official == "val"), np.flatnonzero(official == "train")),
        "aleatorio por imagen (test vs train)": (rand_test, rand_train),
        "NUESTRO split por bloques (test vs train)": (np.flatnonzero(meta.split == "test"),
                                                      np.flatnonzero(meta.split == "train")),
    }
    print("\n=== AUDITORÍA DE FUGA: distancia a la imagen de train más parecida ===")
    print("(más alta = menos fuga; referencia entre imágenes al azar: ~0.14-0.25)")
    for name, (held, tr) in sets.items():
        d = nn_dist(T, held, tr)
        print(f"  {name:<44} mediana {np.median(d):.3f} | p10 {np.percentile(d, 10):.3f}")

    # ---- Guardado ----
    meta["official_split"] = official
    meta["file"] = [images[s].relative_to(root).as_posix() for s in meta.stem]
    meta[["stem", "file", "official_split", "group", "split"]].to_csv(
        out / "data" / "splits" / "splits.csv", index=False)
    for s in SPLITS:
        (out / "data" / "splits" / f"{s}.txt").write_text(
            "\n".join(meta[meta.split == s].file) + "\n")
    json.dump(dict(seed=args.seed, chunk=args.chunk, min_ann=args.min_ann,
                   target=TARGET.tolist(), n_classes=nc, class_names=names,
                   rare_classes_train_only=[int(i) for i in np.flatnonzero(rare)]),
              open(out / "data" / "splits" / "split_config.json", "w"), indent=2)

    ax = tab[SPLITS].plot(kind="bar", stacked=True, figsize=(12, 5))
    ax.set_yscale("log")
    ax.set_xticks(range(nc))
    ax.set_xticklabels([f"{i}:{n}" for i, n in enumerate(names)], rotation=90)
    ax.set_ylabel("anotaciones (escala log)")
    ax.set_title("Distribución de clases en nuestras particiones")
    plt.tight_layout()
    plt.savefig(out / "figures" / "split_distribution.png", dpi=150)
    print("\nListo: data/splits/{train,val,test}.txt, splits.csv, split_config.json, "
          "results/split_class_table.csv, figures/split_distribution.png")


if __name__ == "__main__":
    main()