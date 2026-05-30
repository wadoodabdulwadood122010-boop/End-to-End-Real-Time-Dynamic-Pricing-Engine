from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Construct absolute paths using the '/' operator
RAW_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'raw.csv'
INTERIM_DATA_PATH = BASE_DIR / 'data' / 'interim' / 'interim.csv'
PROCESSED_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'processed.csv'
SPLIT_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'splited' 
MODEL_PATH = BASE_DIR / 'model' / 'model.pkl'