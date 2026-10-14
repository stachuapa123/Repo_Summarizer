from pathlib import Path
folder_path = "/home/stachuapa123/Desktop/MCP_Project"

ROOT = Path(folder_path).expanduser().resolve()
SKIPPED_EXTENTIONS = {".git", "node_modules", "__pycache__", ".venv", "venv"}

def safe_path(path: str) -> Path:
    """Turn the model's path into a real path, refusing anything outside ROOT."""
    full = (ROOT / path).resolve()
    if not full.is_relative_to(ROOT):
        raise ValueError(f"'{path}' is outside the project folder")
    return full

def list_files(path: str = ".", pattern: str = "*", max_results: int = 200) -> str:
    base = safe_path(path)
    if not base.is_dir():
        return f"Not a folder: {path}"
    found_files = []
    for folder, subfolders, files in base.walk():
        subfolders[:] = sorted(d for d in subfolders if d not in SKIPPED_EXTENTIONS)
        for name in sorted(files):
            if Path(name).match(pattern):
                found_files.append((folder / name).relative_to(ROOT).as_posix())
                if len(found_files) >= max_results:
                    return "\n".join(found_files) + f"stopped at {max_results}"
    return "\n".join(found_files) or "(no matching files)"

if __name__ == "__main__":
    sigma_files = list_files(folder_path)
    print(sigma_files)