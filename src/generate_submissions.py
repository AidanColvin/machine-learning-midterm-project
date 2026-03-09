import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

SUBMISSIONS_DIR = Path("data/submissions")
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    X_train_spl = pd.read_csv('data/preprocessed/X_train_spline.csv')
    X_train_las = pd.read_csv('data/preprocessed/X_train_lasso.csv')
    y_train     = pd.read_csv('data/preprocessed/y_train.csv')['Heart Disease']

    test_df     = pd.read_csv('data/raw/test.csv')
    test_df     = test_df.drop(columns=[c for c in ['id','is_outlier'] if c in test_df.columns])

    # align test columns to lasso and spline feature sets
    X_test_las  = test_df.reindex(columns=X_train_las.columns, fill_value=0)
    X_test_spl  = test_df.reindex(columns=X_train_spl.columns, fill_value=0)

    ids = pd.read_csv('data/raw/test.csv')['id'] if 'id' in pd.read_csv('data/raw/test.csv', nrows=1).columns else pd.RangeIndex(len(test_df))
    return X_train_spl, X_train_las, y_train, X_test_spl, X_test_las, ids

def save_submission(probs, ids, name):
    path = SUBMISSIONS_DIR / f"submission_{name}.csv"
    pd.DataFrame({'id': ids, 'Heart Disease': probs}).to_csv(path, index=False)
    print(f"  Saved → {path}")

def run():
    print("\n  ▶  Generating Submissions")
    print("  ══════════════════════════════════════════")
    X_tr_spl, X_tr_las, y_train, X_te_spl, X_te_las, ids = load_data()

    # Logistic Regression Ridge + Spline
    print("\n  ── Logistic Regression Ridge + Spline ──")
    lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=3000, random_state=42)
    lr.fit(X_tr_spl, y_train)
    save_submission(lr.predict_proba(X_te_spl)[:, 1], ids, 'logistic_ridge_spline')

    # Random Forest
    print("\n  ── Random Forest ───────────────────────")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=10,
                                max_features='sqrt', random_state=42, n_jobs=1)
    rf.fit(X_tr_las, y_train)
    save_submission(rf.predict_proba(X_te_las)[:, 1], ids, 'random_forest')

    # Gradient Boosting
    print("\n  ── Gradient Boosting ───────────────────")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, random_state=42)
    gb.fit(X_tr_las, y_train)
    save_submission(gb.predict_proba(X_te_las)[:, 1], ids, 'gradient_boosting')

    # Ensemble
    print("\n  ── Ensemble (avg all 3) ────────────────")
    ensemble = (lr.predict_proba(X_te_spl)[:, 1] +
                rf.predict_proba(X_te_las)[:, 1] +
                gb.predict_proba(X_te_las)[:, 1]) / 3
    save_submission(ensemble, ids, 'ensemble_all')

    print("\n  ✔  Done\n")

if __name__ == "__main__":
    run()
