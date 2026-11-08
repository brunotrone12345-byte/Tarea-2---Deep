import argparse
from pathlib import Path

ap = argparse.ArgumentParser(description="Verifica etiquetas sin imagen asociada en el dataset original.")
ap.add_argument("--root", default="data/raw/FjordFish", help="Ruta al dataset FjordFish original")
root = Path(ap.parse_args().root)
for s in ("train", "val"):
    if not (root / "images" / s).exists():
        continue
    stems = {p.stem for p in (root / "images" / s).iterdir()}
    for l in (root / "labels" / s).glob("*.txt"):
        if l.stem not in stems:
            print(s, l.name)
            print(l.read_text())