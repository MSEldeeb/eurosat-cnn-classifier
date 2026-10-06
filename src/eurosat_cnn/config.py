"""Project-wide constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

IMG_SIZE = 64          # EuroSAT RGB tiles are 64 x 64 pixels (10 m per pixel)
NUM_CHANNELS = 3
SEED = 42

# Fraction of the data held out for validation and for the final test.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
