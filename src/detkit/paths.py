"""
paths.py - Resolución portable de rutas del dataset (Windows, Linux, macOS y Google Colab).

Permite que `data/fjord/data.yaml` use una ruta relativa (`path: data/fjord`) y funcione igual
sin importar desde qué directorio o sistema operativo se ejecuten los scripts o Ultralytics.
"""
from pathlib import Path
import yaml


def resolve_dataset_cfg(yaml_path):
    """Devuelve (root_path, class_names, raw_cfg) resolviendo `path` de forma portable."""
    yp = Path(yaml_path).resolve()
    with open(yp, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

    raw_p = str(cfg.get("path", "")).strip()
    # Normalizar separadores Windows si se abre en Linux/Colab
    norm_p = raw_p.replace("\\", "/")
    candidates = []
    if raw_p:
        candidates.append(Path(raw_p))
        candidates.append(Path(norm_p))
        # Si se ejecuta desde un subdirectorio (ej. notebooks/ o src/), probar raíz del repo
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / norm_p)
    candidates.append(yp.parent)

    root = None
    for c in candidates:
        try:
            if (c / "images").is_dir():
                root = c.resolve()
                break
        except OSError:
            continue
    if root is None:
        root = (Path(norm_p) if norm_p else yp.parent).resolve()
    return root, names, cfg


def write_resolved_yaml(yaml_path, dst_path):
    """Escribe una copia de data.yaml con `path` absoluto local para Ultralytics YOLO."""
    root, _, cfg = resolve_dataset_cfg(yaml_path)
    out_cfg = dict(cfg)
    out_cfg["path"] = str(root)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False, allow_unicode=True)
    return dst.resolve()
