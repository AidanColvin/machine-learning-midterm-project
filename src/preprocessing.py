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

#####################################################################################################################
# PREPROCESSING 
#####################################################################################################################

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats



PREPROCESSED_DIR  = Path("data/preprocessed")
PREPROCESSED_FILE = PREPROCESSED_DIR / "preprocessed-train-data.csv"


TARGET_COLUMN = "Heart Disease"
ID_COLUMN     = "id"

CONTINUOUS_COLUMNS = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
BINARY_COLUMNS     = ["Sex", "FBS over 120", "Exercise angina"]
ORDINAL_COLUMNS    = ["Chest pain type", "EKG results", "Slope of ST",
                      "Number of vessels fluro", "Thallium"]

EXPECTED_DTYPES: dict[str, type] = {
    "Age"                    : float,
    "BP"                     : float,
    "Cholesterol"            : float,
    "Max HR"                 : float,
    "ST depression"          : float,
    "Sex"                    : int,
    "FBS over 120"           : int,
    "Exercise angina"        : int,
    "Chest pain type"        : int,
    "EKG results"            : int,
    "Slope of ST"            : int,
    "Number of vessels fluro": int,
    "Thallium"               : int,
}

OUTLIER_ZSCORE_THRESHOLD    = 3.0
CLASS_BALANCE_WARN_THRESHOLD = 0.30


#####################################################################################################################
# DROP ID
#####################################################################################################################

def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    given the raw dataframe
    return dataframe with the id column removed
    id carries no predictive signal
    """
    return df.drop(columns=[ID_COLUMN])


#####################################################################################################################
# ENCODE TARGET
#####################################################################################################################

def map_presence_absence_to_binary(series: pd.Series) -> pd.Series:
    """
    given a series of 'Presence'/'Absence' strings
    return an integer series of 1 (Presence) and 0 (Absence)
    """
    return series.map({"Presence": 1, "Absence": 0})


def encode_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe with a string-valued Heart Disease column
    return dataframe with that column replaced by binary integers
    """
    df = df.copy()
    df[TARGET_COLUMN] = map_presence_absence_to_binary(df[TARGET_COLUMN])
    return df


#####################################################################################################################
# COERCE TYPES
#####################################################################################################################

def coerce_column_to_dtype(series: pd.Series, dtype: type) -> pd.Series:
    """
    given a series and a target dtype
    return the series cast to that dtype
    """
    return series.astype(dtype)


def coerce_all_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with each column cast to its expected dtype
    prevents silent type errors in downstream statistical models
    """
    df = df.copy()
    for col, dtype in EXPECTED_DTYPES.items():
        if col in df.columns:
            df[col] = coerce_column_to_dtype(df[col], dtype)
    return df


#####################################################################################################################
# IMPUTE MISSING
#####################################################################################################################

def compute_column_median(series: pd.Series) -> float:
    """
    given a numeric series
    return its median ignoring nulls
    """
    return series.median()


def compute_column_mode(series: pd.Series) -> object:
    """
    given a series
    return the most frequent value ignoring nulls
    """
    return series.mode()[0]


def fill_nulls_with_value(series: pd.Series, value: object) -> pd.Series:
    """
    given a series and a scalar fill value
    return the series with all nulls replaced by that value
    """
    return series.fillna(value)


def impute_single_continuous_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    given a dataframe and a continuous column name
    return dataframe with that column's nulls filled by its median
    median is robust to outliers — preferred in clinical biostatistics
    """
    df = df.copy()
    df[col] = fill_nulls_with_value(df[col], compute_column_median(df[col]))
    return df


def impute_single_categorical_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    given a dataframe and a categorical column name
    return dataframe with that column's nulls filled by its mode
    """
    df = df.copy()
    df[col] = fill_nulls_with_value(df[col], compute_column_mode(df[col]))
    return df


def impute_all_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with all continuous columns median-imputed
    """
    for col in CONTINUOUS_COLUMNS:
        if col in df.columns:
            df = impute_single_continuous_column(df, col)
    return df


def impute_all_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with all binary and ordinal columns mode-imputed
    """
    for col in BINARY_COLUMNS + ORDINAL_COLUMNS:
        if col in df.columns:
            df = impute_single_categorical_column(df, col)
    return df


#####################################################################################################################
# DROP ZERO VARIANCE
#####################################################################################################################

def compute_column_variance(series: pd.Series) -> float:
    """
    given a numeric series
    return its variance
    """
    return series.var()


def is_zero_variance(series: pd.Series) -> bool:
    """
    given a numeric series
    return True if variance is zero
    zero-variance columns carry no information and break some models
    """
    return compute_column_variance(series) == 0.0


def get_zero_variance_columns(df: pd.DataFrame) -> list[str]:
    """
    given a dataframe
    return list of numeric column names where variance is zero
    """
    return [col for col in df.select_dtypes(include=[np.number]).columns
            if is_zero_variance(df[col])]


def drop_zero_variance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with all zero-variance columns removed
    """
    return df.drop(columns=get_zero_variance_columns(df))


#####################################################################################################################
# FLAG OUTLIERS
#####################################################################################################################

def compute_absolute_zscores(series: pd.Series) -> pd.Series:
    """
    given a numeric series
    return absolute z-scores aligned to the original index
    """
    z = np.abs(stats.zscore(series.dropna()))
    return pd.Series(z, index=series.dropna().index).reindex(series.index)


def flag_single_column_outliers(series: pd.Series) -> pd.Series:
    """
    given a numeric series
    return a boolean series True where z-score exceeds threshold
    """
    return compute_absolute_zscores(series) > OUTLIER_ZSCORE_THRESHOLD


def build_per_column_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return a boolean dataframe with one column per continuous feature
    """
    return pd.DataFrame(
        {col: flag_single_column_outliers(df[col])
         for col in CONTINUOUS_COLUMNS if col in df.columns},
        index=df.index
    )


def combine_flags_to_row_level(flag_df: pd.DataFrame) -> pd.Series:
    """
    given a boolean dataframe of per-column outlier flags
    return a single boolean series True if any column flagged that row
    """
    return flag_df.any(axis=1)


def flag_outlier_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with a new 'is_outlier' boolean column
    outliers are flagged for transparent reporting, not silently removed
    """
    df = df.copy()
    per_column_flags = build_per_column_outlier_flags(df)
    df["is_outlier"]  = combine_flags_to_row_level(per_column_flags)
    return df


#####################################################################################################################
# CLASS BALANCE
#####################################################################################################################

def compute_class_proportions(series: pd.Series) -> pd.Series:
    """
    given the binary target series
    return a series of class proportions summing to 1.0
    """
    return series.value_counts(normalize=True).sort_index()


def get_minority_class_proportion(series: pd.Series) -> float:
    """
    given the binary target series
    return the proportion of the minority class
    """
    return compute_class_proportions(series).min()


def is_class_imbalanced(series: pd.Series) -> bool:
    """
    given the binary target series
    return True if minority class proportion is below the warn threshold
    """
    return get_minority_class_proportion(series) < CLASS_BALANCE_WARN_THRESHOLD


def get_class_balance_report(series: pd.Series) -> dict[str, object]:
    """
    given the binary target series
    return a dict with counts, proportions, and an imbalance flag
    use this output in your dataset description section
    """
    return {
        "counts"              : series.value_counts().sort_index().to_dict(),
        "proportions"         : compute_class_proportions(series).to_dict(),
        "imbalanced"          : is_class_imbalanced(series),
        "minority_proportion" : get_minority_class_proportion(series),
    }


#####################################################################################################################
# STANDARDIZE
#####################################################################################################################

def compute_column_mean(series: pd.Series) -> float:
    """
    given a numeric series
    return its mean
    """
    return series.mean()


def compute_column_std(series: pd.Series) -> float:
    """
    given a numeric series
    return its standard deviation
    """
    return series.std()


def zscore_scale_series(series: pd.Series) -> pd.Series:
    """
    given a numeric series from training data only
    return it rescaled to mean 0 and standard deviation 1
    parameters must never be fit on test data
    """
    return (series - compute_column_mean(series)) / compute_column_std(series)


def standardize_single_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    given a dataframe and a continuous column name
    return dataframe with that column z-score standardized
    """
    df      = df.copy()
    df[col] = zscore_scale_series(df[col])
    return df


def standardize_all_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe
    return dataframe with all continuous columns z-score standardized
    binary and ordinal columns are untouched — their scales are meaningful
    """
    for col in CONTINUOUS_COLUMNS:
        if col in df.columns:
            df = standardize_single_column(df, col)
    return df


#####################################################################################################################
# SAVE
#####################################################################################################################

def create_output_directory(path: Path) -> None:
    """
    given a directory path
    create it and any missing parents if it does not exist
    """
    path.mkdir(parents=True, exist_ok=True)


def write_dataframe_to_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    given a dataframe and a destination filepath
    write it to csv without the row index
    """
    df.to_csv(filepath, index=False)


def save_preprocessed_data(df: pd.DataFrame) -> None:
    """
    given the fully preprocessed dataframe
    write it to data/preprocessed/preprocessed-train-data.csv
    """
    create_output_directory(PREPROCESSED_DIR)
    write_dataframe_to_csv(df, PREPROCESSED_FILE)


#####################################################################################################################
# PIPELINE
#####################################################################################################################

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    given the raw training dataframe
    return (preprocessed_df, class_balance_report)
    also writes the full frame to data/preprocessed/preprocessed-train-data.csv
    """
    df = drop_id_column(df)
    df = encode_target_column(df)
    df = coerce_all_dtypes(df)
    df = impute_all_continuous(df)
    df = impute_all_categorical(df)
    df = drop_zero_variance_columns(df)
    df = flag_outlier_rows(df)

    balance_report = get_class_balance_report(df[TARGET_COLUMN])

    df = standardize_all_continuous(df)

    save_preprocessed_data(df)

    return df, balance_report


if __name__ == "__main__":
    df_clean, balance_report = preprocess(df_raw)
    print("Preprocessing complete.")
