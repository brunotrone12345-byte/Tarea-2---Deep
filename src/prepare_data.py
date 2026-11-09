"""
prepare_data.py - Paso 1.7: construye el dataset de trabajo con NUESTRAS particiones.

Qué hace:
  - Lee data/splits/splits.csv (creado por make_splits.py).
  - Redimensiona cada imagen para que su lado mayor sea <= --max-side (sin agrandar,
    sin deformar). Las etiquetas YOLO están normalizadas (0-1), así que NO cambian.
  - Deja todo en data/fjord/{images,labels}/{train,val,test}/ y crea data/fjord/data.yaml
    (formato Ultralytics) con los nombres de clase limpios.
  - No modifica el dataset original.

Uso:
    python src/prepare_data.py --root "RUTA/A/FjordFish"
"""
import argparse
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

NAME_FIX = {"gadines_unknown": "gadids_unknown"}   # typo del data.yaml original


def resize_one(args):
    src, dst, max_side = args
    with Image.open(src) as im:
        W, H = im.size
        scale = max_side / max(W, H)
        if scale < 1:
            im.draft("RGB", (int(W * scale) * 2, int(H * scale) * 2))  # decodificación rápida
            im = im.convert("RGB").resize((round(W * scale), round(H * scale)), Image.LANCZOS)
        else:
            im = im.convert("RGB")
        im.save(dst, quality=95)
    return Image.open(dst).size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    dst_root = out / "data" / "fjord"

    # etiquetas indexadas por nombre (la de 2515 está en otra carpeta oficial)
    labels = {p.stem: p for d in (root / "labels").iterdir() if d.is_dir()
              for p in d.glob("*.txt")}
    sp = pd.read_csv(out / "data" / "splits" / "splits.csv", dtype={"stem": str})

    jobs = []
    for r in sp.itertuples():
        for kind in ("images", "labels"):
            (dst_root / kind / r.split).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(labels[r.stem], dst_root / "labels" / r.split / f"{r.stem}.txt")
        jobs.append((root / r.file, dst_root / "images" / r.split / f"{r.stem}.jpg", args.max_side))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        sizes = list(tqdm(ex.map(resize_one, jobs, chunksize=8), total=len(jobs), desc="Redimensionando"))
    sp["new_W"], sp["new_H"] = zip(*sizes)
    sp.to_csv(out / "data" / "splits" / "splits_resized.csv", index=False)

    cfg = yaml.safe_load(open(root / "data.yaml", encoding="utf-8"))
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    names = [NAME_FIX.get(n, n) for n in (re.sub(r"^\d+\s*:\s*", "", str(n)).strip() for n in names)]
    data = {"path": str(dst_root.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "nc": len(names), "names": {i: n for i, n in enumerate(names)}}
    with open(dst_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print("\n=== VERIFICACIÓN ===")
    for s in ("train", "val", "test"):
        ni = len(list((dst_root / "images" / s).glob("*.jpg")))
        nl = len(list((dst_root / "labels" / s).glob("*.txt")))
        nb = sum(len([l for l in p.read_text().splitlines() if l.strip()])
                 for p in (dst_root / "labels" / s).glob("*.txt"))
        print(f"  {s:<5} imágenes={ni}  etiquetas={nl}  cajas={nb}")
    print(f"Tamaño tras redimensionar: lado mayor máx = {max(sp.new_W.max(), sp.new_H.max())} px")
    print("Primeros nombres de clase:", names[:6], "...")
    print(f"Listo: {dst_root / 'data.yaml'}")


if __name__ == "__main__":
    main()