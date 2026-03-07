#####################################################################################################################
# IMPORTS 
#####################################################################################################################

import pandas as pd
from pathlib import Path


#####################################################################################################################
# LOADING RAW TRAINING DATA
#####################################################################################################################

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
TRAIN_FILE = RAW_DATA_DIR / "train.csv"

def build_file_path(directory: Path, filename: str) -> Path:
    """
    given a directory path and a filename
    return the full joined path
    """
    return directory / filename


def validate_file_exists(filepath: Path) -> Path:
    """
    given a file path
    return the same path if the file exists
    raises FileNotFoundError if it does not
    """
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at: {filepath}")
    return filepath


def read_csv_to_dataframe(filepath: Path) -> pd.DataFrame:
    """
    given a validated file path to a csv
    return its contents as a dataframe
    """
    return pd.read_csv(filepath)


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    given a path to a csv file
    return its contents as a dataframe
    validates existence before reading
    """
    valid_path = validate_file_exists(filepath)
    return read_csv_to_dataframe(valid_path)


def load_train_data() -> pd.DataFrame:
    """
    given nothing
    return the raw training dataframe from data/raw/train.csv
    """
    filepath = build_file_path(RAW_DATA_DIR, "train.csv")
    return load_csv(filepath)

if __name__ == "__main__":
    df = load_train_data()
    print(f"Raw training data loaded: {len(df):,} rows × {len(df.columns)} columns")