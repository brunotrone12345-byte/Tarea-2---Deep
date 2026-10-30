from pathlib import Path
root = Path(r"C:\Users\bruno\Documents\Code\Tarea 2 - Deep\data\raw\FjordFish")   # cambia la ruta
for s in ("train", "val"):
    stems = {p.stem for p in (root / "images" / s).iterdir()}
    for l in (root / "labels" / s).glob("*.txt"):
        if l.stem not in stems:
            print(s, l.name); print(l.read_text())