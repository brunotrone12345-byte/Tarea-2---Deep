from pathlib import Path, PureWindowsPath


def resolve_dataset_root(yaml_path, cfg_path):
    """Resuelve la raíz del dataset de forma portable entre equipos/OS."""
    yaml_path = Path(yaml_path).resolve()
    direct = Path(cfg_path)
    if direct.exists():
        return direct.resolve()
    if not direct.is_absolute():
        rel = (yaml_path.parent / direct).resolve()
        if rel.exists():
            return rel
    win = PureWindowsPath(str(cfg_path))
    if win.drive:
        tail = Path(*win.parts[1:]).resolve()
        if tail.exists():
            return tail
    return yaml_path.parent
