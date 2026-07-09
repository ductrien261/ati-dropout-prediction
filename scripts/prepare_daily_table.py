import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ati.data.prepare import prepare_daily_table

if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "oulad"
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
    prepare_daily_table(DATA_DIR, OUTPUT_DIR)