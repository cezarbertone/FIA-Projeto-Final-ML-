from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataPipeline.config import LANDING_DIR, RAW_FILES, find_landing_file


def run():
    print(f"Landing: {LANDING_DIR}")
    found = {}
    for logical_name in RAW_FILES:
        path = find_landing_file(logical_name)
        found[logical_name] = str(path)
        print(f"OK {logical_name}: {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    return found


if __name__ == "__main__":
    run()
