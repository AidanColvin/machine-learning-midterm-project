#####################################################################################################################
# IMPORTS
#####################################################################################################################

import pandas as pd
from pathlib import Path


#####################################################################################################################
# CONFIG
#####################################################################################################################

DATA_FILE       = Path("data/preprocessed/preprocessed-train-data.csv")
TARGET_COLUMN   = "Heart Disease"
IGNORE_COLUMNS  = ["is_outlier"]

CONTINUOUS_COLUMNS = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
BINARY_COLUMNS     = ["Sex", "FBS over 120", "Exercise angina"]
ORDINAL_COLUMNS    = ["Chest pain type", "EKG results", "Slope of ST",
                      "Number of vessels fluro", "Thallium"]


#####################################################################################################################
# LOAD
#####################################################################################################################

def load_dataframe(filepath: Path) -> pd.DataFrame:
    """
    given a path to a csv file
    return its contents as a dataframe
    raises FileNotFoundError if file does not exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at: {filepath}")
    return pd.read_csv(filepath)


#####################################################################################################################
# SAMPLE SIZE
#####################################################################################################################

def get_sample_size(df: pd.DataFrame) -> int:
    """
    given a dataframe
    return the number of rows
    """
    return len(df)


#####################################################################################################################
# FEATURES
#####################################################################################################################

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    given a dataframe
    return list of column names excluding the target and ignored columns
    """
    excluded = [TARGET_COLUMN] + IGNORE_COLUMNS
    return [col for col in df.columns if col not in excluded]


def get_feature_count(df: pd.DataFrame) -> int:
    """
    given a dataframe
    return the number of predictor features
    """
    return len(get_feature_columns(df))


def get_feature_type(col: str) -> str:
    """
    given a column name
    return its type label: continuous, binary, or ordinal
    """
    if col in CONTINUOUS_COLUMNS:
        return "continuous"
    if col in BINARY_COLUMNS:
        return "binary"
    if col in ORDINAL_COLUMNS:
        return "ordinal"
    return "unknown"


#####################################################################################################################
# DESCRIPTIVE STATISTICS
#####################################################################################################################

def get_continuous_stats(df: pd.DataFrame, col: str) -> dict:
    """
    given a dataframe and a continuous column name
    return a dict of mean, std, min, and max
    """
    return {
        "mean": df[col].mean(),
        "std" : df[col].std(),
        "min" : df[col].min(),
        "max" : df[col].max(),
    }


def get_categorical_stats(df: pd.DataFrame, col: str) -> dict:
    """
    given a dataframe and a categorical column name
    return a dict of mode and unique value count
    """
    return {
        "mode"         : df[col].mode()[0],
        "unique_values": df[col].nunique(),
    }


def get_column_missing_count(df: pd.DataFrame, col: str) -> int:
    """
    given a dataframe and a column name
    return the number of missing values in that column
    """
    return df[col].isna().sum()


#####################################################################################################################
# RESPONSE VARIABLE
#####################################################################################################################

def get_response_variable_summary(df: pd.DataFrame) -> dict:
    """
    given a dataframe
    return a dict describing the target column distribution
    """
    counts      = df[TARGET_COLUMN].value_counts().sort_index()
    proportions = df[TARGET_COLUMN].value_counts(normalize=True).sort_index()
    return {
        "name"       : TARGET_COLUMN,
        "type"       : "binary",
        "classes"    : {0: "Absence", 1: "Presence"},
        "counts"     : counts.to_dict(),
        "proportions": proportions.to_dict(),
    }


#####################################################################################################################
# PASS / FAIL CHECKS
#####################################################################################################################

def check_file_exists(filepath: Path) -> tuple[str, bool]:
    """
    given a file path
    return (filename, passed) where passed is True if the file exists
    """
    return filepath.name, filepath.exists()


def check_target_column_present(df: pd.DataFrame) -> tuple[str, bool]:
    """
    given a dataframe
    return (check_name, passed) where passed is True if target column exists
    """
    return "target column present", TARGET_COLUMN in df.columns


def check_no_missing_values(df: pd.DataFrame) -> tuple[str, bool]:
    """
    given a dataframe
    return (check_name, passed) where passed is True if no nulls exist
    """
    return "no missing values", df[get_feature_columns(df)].isna().sum().sum() == 0


def check_minimum_sample_size(df: pd.DataFrame, minimum: int = 100) -> tuple[str, bool]:
    """
    given a dataframe and a minimum row count
    return (check_name, passed) where passed is True if sample size meets minimum
    """
    return f"sample size >= {minimum}", get_sample_size(df) >= minimum


def run_all_checks(df: pd.DataFrame, filepath: Path) -> list[tuple[str, bool]]:
    """
    given a dataframe and its filepath
    return list of (check_name, passed) tuples for all validation checks
    """
    return [
        check_file_exists(filepath),
        check_target_column_present(df),
        check_no_missing_values(df),
        check_minimum_sample_size(df),
    ]


#####################################################################################################################
# PRINT REPORT
#####################################################################################################################

def print_section(title: str) -> None:
    """
    given a section title
    print a formatted section header
    """
    print("")
    print(f"  ── {title} {'─' * (44 - len(title))}")


def print_checks(checks: list[tuple[str, bool]]) -> None:
    """
    given a list of (check_name, passed) tuples
    print each check with a PASS or FAIL label
    """
    for name, passed in checks:
        label = "\033[0;32mPASS\033[0m" if passed else "\033[0;31mFAIL\033[0m"
        print(f"     [{label}]  {name}")


def print_feature_stats(df: pd.DataFrame) -> None:
    """
    given a dataframe
    print descriptive statistics for each feature column
    """
    for col in get_feature_columns(df):
        ftype   = get_feature_type(col)
        missing = get_column_missing_count(df, col)

        if ftype == "continuous":
            stats = get_continuous_stats(df, col)
            print(f"     {col:<28} {ftype:<12} "
                  f"mean={stats['mean']:>7.2f}  std={stats['std']:>6.2f}  "
                  f"min={stats['min']:>7.2f}  max={stats['max']:>7.2f}  "
                  f"missing={missing}")
        else:
            stats = get_categorical_stats(df, col)
            print(f"     {col:<28} {ftype:<12} "
                  f"mode={stats['mode']}  unique={stats['unique_values']}  "
                  f"missing={missing}")


def print_response_summary(summary: dict) -> None:
    """
    given a response variable summary dict
    print a formatted class distribution
    """
    print(f"     Name   : {summary['name']}")
    print(f"     Type   : {summary['type']}")
    for cls, label in summary["classes"].items():
        count = summary["counts"].get(cls, 0)
        prop  = summary["proportions"].get(cls, 0)
        bar   = "█" * int(prop * 30)
        print(f"     {label:<10} {bar:<30}  {count:>4} rows  ({prop:.1%})")


#####################################################################################################################
# MAIN
#####################################################################################################################

if __name__ == "__main__":
    print("")
    print("  ▶  Dataset Description Report")
    print("  ══════════════════════════════════════════════")

    df = load_dataframe(DATA_FILE)

    print_section("File Checks")
    checks = run_all_checks(df, DATA_FILE)
    print_checks(checks)

    print_section("Overview")
    print(f"     Sample size      : {get_sample_size(df):,} rows")
    print(f"     Number of features: {get_feature_count(df)}")
    print(f"     Response variable : {TARGET_COLUMN}")

    print_section("Response Variable")
    print_response_summary(get_response_variable_summary(df))

    print_section("Feature Predictors")
    print(f"     {'Feature':<28} {'Type':<12} Statistics")
    print(f"     {'───────':<28} {'────':<12} ──────────")
    print_feature_stats(df)

    print("")
    print("  ══════════════════════════════════════════════")
    all_passed = all(passed for _, passed in checks)
    if all_passed:
        print("  \033[0;32m✔  All checks passed.\033[0m")
    else:
        print("  \033[0;31m✘  Some checks failed — review above.\033[0m")
    print("")