"""
build_groups.py - Paso 1.5: agrupar imágenes consecutivas en "escenas".

Idea: las imágenes salen de videos, así que IDs vecinos suelen ser casi el mismo
fotograma. Para no repartir una misma escena entre train y test, agrupamos y luego
repartiremos GRUPOS completos (no imágenes sueltas).

Requiere haber ejecutado antes explore_dataset.py (usa results/dataset_images.csv).

Uso:
    python src/build_groups.py                  # usa el umbral automático
    python src/build_groups.py --thr-pct 10     # umbral más estricto (más grupos)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

THUMB = (32, 18)


def thumb(path):
    """Miniatura 32x18 en color. draft() hace que el JPEG se decodifique ya reducido (rápido)."""
    with Image.open(path) as im:
        im.draft("RGB", (THUMB[0] * 8, THUMB[1] * 8))
        im = im.convert("RGB").resize(THUMB, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).ravel() / 255.0


def make_groups(d_cons, same_res, thr, max_len):
    """Corta el grupo si: la imagen cambia mucho, cambia la resolución o el grupo es muy largo."""
    g, cur_len = [0], 1
    for i in range(len(d_cons)):
        if d_cons[i] > thr or not same_res[i] or cur_len >= max_len:
            g.append(g[-1] + 1)
            cur_len = 1
        else:
            g.append(g[-1])
            cur_len += 1
    return np.array(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".", help="carpeta del proyecto")
    ap.add_argument("--thr-pct", type=float, default=25,
                    help="umbral = este percentil de las distancias entre imágenes AL AZAR")
    ap.add_argument("--max-len", type=int, default=60, help="máximo de imágenes por grupo")
    args = ap.parse_args()
    out = Path(args.out)
    res_dir, fig_dir = out / "results", out / "figures"

    imgs = pd.read_csv(res_dir / "dataset_images.csv", dtype={"stem": str})
    imgs["id"] = imgs.stem.astype(int)
    imgs = imgs.sort_values("id").reset_index(drop=True)

    cache = res_dir / "thumbs.npy"
    if cache.exists() and np.load(cache).shape[0] == len(imgs):
        T = np.load(cache)
    else:
        T = np.stack([thumb(f) for f in tqdm(imgs.file, desc="Miniaturas")])
        np.save(cache, T)

    d_cons = np.abs(T[1:] - T[:-1]).mean(axis=1)           # distancia entre IDs vecinos
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(T), 5000), rng.integers(0, len(T), 5000)
    d_rand = np.abs(T[a] - T[b]).mean(axis=1)               # distancia entre imágenes al azar

    print("\n=== DISTANCIAS (menor = más parecidas) ===")
    q = [5, 10, 25, 50, 75]
    print("IDs vecinos :", dict(zip(q, np.percentile(d_cons, q).round(3))))
    print("Al azar     :", dict(zip(q, np.percentile(d_rand, q).round(3))))
    print("Si las de IDs vecinos son mucho menores, los IDs siguen el orden de los videos.")

    same_res = ((imgs.W.values[1:] == imgs.W.values[:-1]) &
                (imgs.H.values[1:] == imgs.H.values[:-1]))
    print("\n=== Nº de grupos según el umbral (percentil de las distancias al azar) ===")
    for p in [5, 10, 25, 50]:
        n = make_groups(d_cons, same_res, np.percentile(d_rand, p), args.max_len).max() + 1
        print(f"  p{p:<2}: {n} grupos")

    thr = np.percentile(d_rand, args.thr_pct)
    imgs["group"] = make_groups(d_cons, same_res, thr, args.max_len)
    sizes = imgs.groupby("group").size()
    print(f"\nUsando p{args.thr_pct:g} (umbral {thr:.3f}): {len(sizes)} grupos")
    print("Tamaño de grupo:", sizes.describe().round(1).to_dict())

    mix = imgs.groupby("group").split.nunique()
    print(f"Grupos con imágenes de train Y val oficiales: {(mix > 1).sum()} de {len(sizes)} "
          f"({(mix > 1).mean():.0%})  <- cada uno es fuga potencial en el split oficial")

    imgs[["stem", "id", "split", "W", "H", "n_boxes", "group"]].to_csv(
        res_dir / "groups.csv", index=False)

    fig, axs = plt.subplots(2, 1, figsize=(12, 7))
    axs[0].plot(imgs.id[1:], d_cons, lw=0.5)
    axs[0].axhline(thr, color="r", ls="--", label="umbral")
    axs[0].set_xlabel("ID de imagen")
    axs[0].set_ylabel("distancia con la imagen anterior")
    axs[0].legend()
    axs[1].hist(d_rand, bins=60, alpha=.6, density=True, label="pares al azar")
    axs[1].hist(d_cons, bins=60, alpha=.6, density=True, label="IDs vecinos")
    axs[1].axvline(thr, color="r", ls="--")
    axs[1].set_xlabel("distancia")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "group_distances.png", dpi=150)
    plt.close()

    # Hoja de contacto: primera / media / última imagen de 5 grupos grandes al azar
    big = sizes[sizes >= 3].index.to_numpy()
    pick = np.random.default_rng(1).choice(big, size=min(5, len(big)), replace=False)
    W = 400
    rows = []
    for g in pick:
        sub = imgs[imgs.group == g]
        tiles = []
        for k in (0, len(sub) // 2, len(sub) - 1):
            with Image.open(sub.iloc[k].file) as im:
                im.draft("RGB", (W * 2, W * 2))
                im = im.convert("RGB").resize((W, int(W * 9 / 16)))
            tiles.append(im)
        rows.append(tiles)
    sheet = Image.new("RGB", (3 * W, len(rows) * int(W * 9 / 16)))
    for r, tiles in enumerate(rows):
        for c, t in enumerate(tiles):
            sheet.paste(t, (c * W, r * int(W * 9 / 16)))
    sheet.save(fig_dir / "group_examples.jpg", quality=88)
    print("\nListo: results/groups.csv, figures/group_distances.png, figures/group_examples.jpg")


if __name__ == "__main__":
    main()