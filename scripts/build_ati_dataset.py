import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ati.data.build_dataset import build_full_dataset

if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "oulad"
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

    build_full_dataset(DATA_DIR, OUTPUT_DIR)