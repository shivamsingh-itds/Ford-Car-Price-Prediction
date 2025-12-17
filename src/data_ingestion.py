# src/data_ingestion.py
from pathlib import Path
import pandas as pd

def load_data():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "raw" / "ford.csv"
    return pd.read_csv(data_path)
