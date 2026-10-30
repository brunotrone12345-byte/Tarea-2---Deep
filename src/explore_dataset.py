"""
explore_dataset.py - Etapa 1: exploración del dataset FjordFish.

Uso (desde la carpeta del proyecto):
    python src/explore_dataset.py --root "RUTA/A/FjordFish"

No modifica el dataset. Escribe CSV en results/ y gráficos en figures/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # guardar figuras sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png"}
# Anotaciones por clase según la Tabla 1 de Zenodo (para contrastar con lo real)
OFFICIAL = [93, 47, 31, 456, 1215, 1433, 152, 333, 811, 181, 7, 149, 22, 39,
            101, 200, 684, 0, 190, 38, 15, 1, 5, 30, 113, 1, 3]
REF_SIZE = 640  # tamaño de referencia para medir qué tan pequeños son los objetos


def load_names(root):
    with open(root / "data.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def scan(root, nc):
    """Recorre images/<split> y labels/<split>; valida cada línea de cada etiqueta."""
    images, boxes, orphans = [], [], []
    splits = sorted(d.name for d in (root / "images").iterdir() if d.is_dir())
    for split in splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        for p in tqdm(files, desc=f"Leyendo {split}"):
            with Image.open(p) as im:  # solo lee la cabecera: rápido
                W, H = im.size
            lbl = lbl_dir / (p.stem + ".txt")
            n_valid = n_bad = 0
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        if len(parts) != 5:
                            raise ValueError
                        c = int(float(parts[0]))
                        xc, yc, w, h = map(float, parts[1:])
                    except ValueError:
                        n_bad += 1
                        continue
                    ok = (0 <= c < nc and 0 < w <= 1 and 0 < h <= 1
                          and 0 <= xc <= 1 and 0 <= yc <= 1)
                    if not ok:
                        n_bad += 1
                        continue
                    n_valid += 1
                    boxes.append(dict(split=split, stem=p.stem, cls=c,
                                      xc=xc, yc=yc, w=w, h=h, W=W, H=H))
            images.append(dict(split=split, stem=p.stem, file=str(p), W=W, H=H,
                               has_label_file=lbl.exists(),
                               n_boxes=n_valid, n_bad=n_bad))
        stems = {p.stem for p in files}
        orphans += [f"{split}/{l.name}" for l in lbl_dir.glob("*.txt")
                    if l.stem not in stems]
    imgs, bxs = pd.DataFrame(images), pd.DataFrame(boxes)
    bxs["key"] = bxs["split"] + "/" + bxs["stem"]
    bxs["size_ref"] = ((bxs.w * bxs.W * bxs.h * bxs.H) ** 0.5
                       * REF_SIZE / bxs[["W", "H"]].max(axis=1))
    return imgs, bxs, orphans


def class_table(boxes, names):
    nc = len(names)
    cnt = (boxes.groupby(["cls", "split"]).size().unstack(fill_value=0)
           .reindex(range(nc), fill_value=0))
    tab = pd.DataFrame({"clase": names})
    tab = tab.join(cnt)
    tab["total"] = cnt.sum(axis=1)
    tab["imgs_con_clase"] = (boxes.groupby("cls").key.nunique()
                             .reindex(range(nc), fill_value=0))
    if len(OFFICIAL) == nc:
        tab["oficial"] = OFFICIAL
        tab["coincide"] = tab["total"] == tab["oficial"]
    return tab


def report(imgs, boxes, orphans, names):
    print("\n=== IMÁGENES Y ETIQUETAS ===")
    print(imgs.groupby("split").agg(
        imagenes=("stem", "count"), cajas=("n_boxes", "sum"),
        sin_cajas=("n_boxes", lambda s: int((s == 0).sum()))).to_string())
    print(f"Total imágenes: {len(imgs)} | Total cajas válidas: {len(boxes)}")
    print("Imágenes sin archivo de etiqueta:", int((~imgs.has_label_file).sum()))
    print("Etiquetas sin imagen:", len(orphans))
    print("Líneas inválidas (formato/rango/clase):", int(imgs.n_bad.sum()))

    print("\n=== RESOLUCIONES (ancho, alto -> nº de imágenes) ===")
    print(imgs.groupby(["W", "H"]).size().sort_values(ascending=False)
          .head(10).to_string())

    print("\n=== CAJAS POR IMAGEN ===")
    print(imgs.n_boxes.describe().round(2).to_string())

    print("\n=== TABLA POR CLASE ===")
    print(class_table(boxes, names).to_string())

    print(f"\n=== TAMAÑO DE OBJETOS (lado equivalente en px, imagen a {REF_SIZE}) ===")
    print(boxes.size_ref.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
    print(f"Objetos < 16 px: {(boxes.size_ref < 16).mean():.1%} | "
          f"< 32 px: {(boxes.size_ref < 32).mean():.1%}")

    print("\n=== PISTAS DE FUGA DE DATOS (data leakage) ===")
    ids = {sp: {int(s) for s in g.stem if s.isdigit()}
           for sp, g in imgs.groupby("split")}
    if "train" in ids and "val" in ids and ids["train"] and ids["val"]:
        tr, va = ids["train"], ids["val"]
        near = sum(1 for i in va if (i - 1) in tr or (i + 1) in tr)
        print(f"Rango de IDs: train {min(tr)}-{max(tr)} | val {min(va)}-{max(va)}")
        print(f"Nombres repetidos entre train y val: {len(tr & va)}")
        print(f"Imágenes de val con vecino (ID±1) en train: {near}/{len(va)} "
              f"({near / len(va):.0%})")


def plots(imgs, boxes, names, fig_dir):
    nc = len(names)
    cnt = (boxes.groupby(["cls", "split"]).size().unstack(fill_value=0)
           .reindex(range(nc), fill_value=0))
    ax = cnt.plot(kind="bar", stacked=True, figsize=(12, 5))
    ax.set_yscale("log")
    ax.set_xticks(range(nc))
    ax.set_xticklabels([f"{i}:{n}" for i, n in enumerate(names)], rotation=90)
    ax.set_ylabel("anotaciones (escala log)")
    ax.set_title("Distribución de clases por partición oficial")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution.png", dpi=150)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(imgs.n_boxes, bins=range(0, int(imgs.n_boxes.max()) + 2))
    axs[0].set_xlabel("cajas por imagen")
    axs[0].set_ylabel("imágenes")
    axs[1].hist(boxes.size_ref.clip(upper=400), bins=50)
    axs[1].set_xlabel(f"tamaño del objeto (px, imagen a {REF_SIZE})")
    axs[1].set_ylabel("cajas")
    plt.tight_layout()
    plt.savefig(fig_dir / "boxes_stats.png", dpi=150)
    plt.close()


def draw_samples(imgs, boxes, names, out, n=6, seed=0):
    """Dibuja las cajas sobre imágenes al azar para verificar que el formato es correcto."""
    cand = imgs[imgs.n_boxes > 0]
    chosen = cand.sample(n=min(n, len(cand)), random_state=seed)
    tiles = []
    for _, r in chosen.iterrows():
        im = Image.open(r.file).convert("RGB")
        im = im.resize((800, round(800 * im.height / im.width)))
        d = ImageDraw.Draw(im)
        sel = boxes[(boxes.split == r.split) & (boxes.stem == r.stem)]
        for _, b in sel.iterrows():
            x1, x2 = (b.xc - b.w / 2) * im.width, (b.xc + b.w / 2) * im.width
            y1, y2 = (b.yc - b.h / 2) * im.height, (b.yc + b.h / 2) * im.height
            d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            d.text((x1 + 2, y1 + 2), f"{int(b.cls)}:{names[int(b.cls)]}",
                   fill=(255, 255, 0))
        d.text((6, 6), f"{r.split}/{r.stem}", fill=(255, 255, 255))
        tiles.append(im)
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    th = max(t.height for t in tiles)
    canvas = Image.new("RGB", (800 * cols, th * rows))
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % cols) * 800, (i // cols) * th))
    canvas.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="carpeta con images/, labels/ y data.yaml")
    ap.add_argument("--out", default=".", help="carpeta del proyecto (results/ y figures/)")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    names = load_names(root)
    print(f"data.yaml: {len(names)} clases")
    imgs, boxes, orphans = scan(root, len(names))
    imgs.to_csv(out / "results" / "dataset_images.csv", index=False)
    boxes.to_csv(out / "results" / "dataset_boxes.csv", index=False)
    class_table(boxes, names).to_csv(out / "results" / "class_table.csv", index=False)
    report(imgs, boxes, orphans, names)
    plots(imgs, boxes, names, out / "figures")
    draw_samples(imgs, boxes, names, out / "figures" / "samples_gt.jpg")
    print("\nListo. Revisa results/ y figures/ (sobre todo figures/samples_gt.jpg).")


if __name__ == "__main__":
    main()