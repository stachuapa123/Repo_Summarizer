from pathlib import Path

FOLDER_PATH = "/home/stachuapa123/Desktop/MCP_Project"
MAX_CHARS = 100_000


class Workspace:
    """File tools confined to one project folder."""

    SKIPPED_EXTENTIONS = {".git", "node_modules", "__pycache__", ".venv", "venv"}

    def __init__(self, root: str | Path, max_chars: int = MAX_CHARS):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Project folder not found: {self.root}")
        self.max_chars = max_chars

    def _safe_path(self, path: str) -> Path:
        full = (self.root / path).resolve()
        if not full.is_relative_to(self.root):
            raise ValueError(f"'{path}' is outside the project folder")
        return full

    def list_files(self, path: str = ".", pattern: str = "*", max_results: int = 200) -> str:
        """List files in the project, skipping .git, node_modules and virtualenvs."""
        base = self._safe_path(path)
        if not base.is_dir():
            raise ValueError(f"Not a folder: {path}")
        found_files = []
        for folder, subfolders, files in base.walk():
            subfolders[:] = sorted(d for d in subfolders if d not in self.SKIPPED_EXTENTIONS)
            for name in sorted(files):
                if Path(name).match(pattern):
                    found_files.append((folder / name).relative_to(self.root).as_posix())
                    if len(found_files) >= max_results:
                        return "\n".join(found_files) + f"\n... (stopped at {max_results})"
        return "\n".join(found_files) or "(no matching files)"

    def read_file(self, path: str) -> str:
        """Read the full text of a file in the project."""
        file = self._safe_path(path)
        if not file.is_file():
            raise ValueError(f"No such file: {path}")
        data = file.read_bytes()
        if b"\x00" in data[:8000]:
            raise ValueError(f"{path} is a binary file, not text")
        text = data.decode("utf-8", errors="replace")
        if len(text) > self.max_chars:
            raise ValueError(f"{path} is too large ({len(text):,} characters, limit {self.max_chars:,})")
        return text


if __name__ == "__main__":
    ws = Workspace(FOLDER_PATH)
    print(ws.list_files("."))
    print("---")
    print(ws.read_file("calculator.py"))