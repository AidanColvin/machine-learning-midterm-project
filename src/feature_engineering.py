import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import SplineTransformer

PREPROCESSED_DIR   = Path("data/preprocessed")
CONTINUOUS_COLUMNS = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

def load_splits():
    X_train = pd.read_csv(PREPROCESSED_DIR / "X_train.csv")
    X_test  = pd.read_csv(PREPROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PREPROCESSED_DIR / "y_train.csv").squeeze()
    y_test  = pd.read_csv(PREPROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

def lasso_feature_selection(X_train, X_test, y_train):
    """
    fit LassoCV on training data only
    drop features where coefficient shrinks to zero
    returns filtered X_train, X_test, and selected column names
    """
    X_tr = X_train.drop(columns=["is_outlier"], errors="ignore")
    X_te = X_test.drop(columns=["is_outlier"],  errors="ignore")

    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_tr, y_train)

    mask     = np.abs(lasso.coef_) > 0
    sel_cols = X_tr.columns[mask].tolist()

    print(f"  Lasso selected {len(sel_cols)}/13 features")
    print(f"  Dropped : {[c for c in X_tr.columns if c not in sel_cols]}")
    print(f"  Kept    : {sel_cols}")

    return X_tr[sel_cols], X_te[sel_cols], sel_cols

def apply_splines(X_train, X_test, sel_cols, n_knots=5, degree=3):
    """
    apply SplineTransformer to continuous features present in sel_cols
    concatenate spline features onto existing dataframe
    fit on training data only, transform both train and test
    """
    cont = [c for c in CONTINUOUS_COLUMNS if c in sel_cols]

    spl     = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
    tr_spl  = spl.fit_transform(X_train[cont])
    te_spl  = spl.transform(X_test[cont])

    n_out   = tr_spl.shape[1]
    per     = n_out // len(cont)
    names   = [f"spl_{c}_{i}" for c in cont for i in range(per)][:n_out]

    X_train_spl = pd.concat([X_train.reset_index(drop=True),
                              pd.DataFrame(tr_spl, columns=names)], axis=1)
    X_test_spl  = pd.concat([X_test.reset_index(drop=True),
                              pd.DataFrame(te_spl, columns=names)], axis=1)

    print(f"  Splines applied to: {cont}")
    print(f"  Features after splines: {X_train_spl.shape[1]}")

    return X_train_spl, X_test_spl

def save_splits(X_train, X_test, y_train, y_test, suffix=""):
    X_train.to_csv(PREPROCESSED_DIR / f"X_train{suffix}.csv", index=False)
    X_test.to_csv( PREPROCESSED_DIR / f"X_test{suffix}.csv",  index=False)
    y_train.to_csv(PREPROCESSED_DIR / f"y_train{suffix}.csv", index=False)
    y_test.to_csv( PREPROCESSED_DIR / f"y_test{suffix}.csv",  index=False)
    print(f"  Saved splits with suffix '{suffix}'")

if __name__ == "__main__":
    print("\n  ▶  Feature Engineering Pipeline")
    print("  ══════════════════════════════════════════")

    X_train, X_test, y_train, y_test = load_splits()

    print("\n  ── Lasso Feature Selection ─────────────")
    X_tr_sel, X_te_sel, sel_cols = lasso_feature_selection(X_train, X_test, y_train)
    save_splits(X_tr_sel, X_te_sel, y_train, y_test, suffix="_lasso")

    print("\n  ── Spline Transformation ───────────────")
    X_tr_spl, X_te_spl = apply_splines(X_tr_sel, X_te_sel, sel_cols)
    save_splits(X_tr_spl, X_te_spl, y_train, y_test, suffix="_spline")

    print("\n  ✔  Done. Saved:")
    print("     data/preprocessed/X_train_lasso.csv")
    print("     data/preprocessed/X_train_spline.csv\n")
