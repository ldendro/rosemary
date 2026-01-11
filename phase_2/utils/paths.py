# phase_2/utils/paths.py
from pathlib import Path

PHASE_2_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PHASE_2_ROOT / "data" / "raw"
PROCESSED_DIR = PHASE_2_ROOT / "data" / "processed"
