import os
from pathlib import Path
from typing import Any, Generator


def traverse_folder(folder: Path) -> Generator[Path, Any, Any]:
    if os.path.exists(folder) and os.path.isdir(folder):
        children = sorted(os.listdir(folder))
        for child in children:
            child_abs_path = Path(os.path.join(folder, child))
            if os.path.isfile(child_abs_path):
                yield child_abs_path
            else:
                yield from traverse_folder(child_abs_path)
    else:
        raise ValueError(f"{folder} does not exist or is not a directory")
