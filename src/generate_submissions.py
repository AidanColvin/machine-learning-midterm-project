import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

CV5  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def print_dataset_stats(X, y, label="Training") -> None:
    print(f"\n  ▶  {label} Dataset Summary")
    print("  ══════════════════════════════════════════════════════")
    print(f"     Sample size        : {len(X):,} rows")
    print(f"     Number of features : {X.shape[1]}")
    print(f"     Dependent variable : Heart Disease (binary)")
    counts = y.value_counts().sort_index()
    for lbl, count in counts.items():
        bar  = "█" * int((count / len(y)) * 20)
        name = "Absence" if lbl == 0 else "Presence"
        print(f"     {name:<10} {bar}  {count:,} ({count/len(y)*100:.1f}%)")
    print(f"\n  ── Feature Descriptive Stats (all {X.shape[1]} features) ────────")
    print(f"     {'Feature':<28} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Missing':>8}")
    print(f"     {'───────':<28} {'────':>8} {'───':>8} {'───':>8} {'───':>8} {'───────':>8}")
    for col in X.columns:
        s = X[col]
        print(f"     {col:<28} {s.mean():>8.3f} {s.std():>8.3f} {s.min():>8.3f} {s.max():>8.3f} {int(s.isna().sum()):>8}")
    print("  ══════════════════════════════════════════════════════\n")

def create_submissions() -> None:
    train_df = pd.read_csv('data/preprocessed/preprocessed-train-data.csv')
    X = train_df.drop(columns=[c for c in ['Heart Disease','id','is_outlier'] if c in train_df.columns])
    y = train_df['Heart Disease']
    if pd.api.types.is_string_dtype(y):
        y = y.map({'Absence': 0, 'Presence': 1}).astype(int)

    test_df  = pd.read_csv('data/raw/test.csv')
    test_ids = test_df['id']
    X_test   = test_df.drop(columns=[c for c in ['id','is_outlier'] if c in test_df.columns])
    X_test   = X_test.reindex(columns=X.columns, fill_value=0)

    print_dataset_stats(X, y, label="Preprocessed Training")

    os.makedirs('data/submissions', exist_ok=True)

    # Already saved logistic + svm — skip those, start from random forest
    models = {
        'random_forest': (
            'Random Forest 100 trees (5-fold)',
            RandomForestClassifier(n_estimators=100, max_depth=15,
                                   min_samples_leaf=10, max_features='sqrt',
                                   random_state=42, n_jobs=1),
            CV5
        ),
        'gradient_boosting': (
            'Gradient Boosting 200 trees (5-fold)',
            GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=42),
            CV5
        ),
        'gradient_boosting_deep': (
            'Gradient Boosting Deep 300 trees (5-fold)',
            GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                       learning_rate=0.03, subsample=0.8,
                                       min_samples_leaf=10, random_state=42),
            CV5
        ),
    }

    trained = {}
    results = []
    for name, (label, model, cv) in models.items():
        pad = '─' * max(1, 52 - len(label))
        print(f"  ── {label} {pad}")
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        auc = cv_scores.mean()
        print(f"     CV AUC : {auc:.4f}  std={cv_scores.std():.4f}  folds={cv.n_splits}")
        print(f"     Training on 100% of data...")
        model.fit(X, y)
        probs = model.predict_proba(X_test)[:, 1]
        out   = f'data/submissions/submission_{name}.csv'
        pd.DataFrame({'id': test_ids, 'Heart Disease': probs}).to_csv(out, index=False)
        print(f"     Saved → {out}\n")
        trained[name] = probs
        results.append({'Model': label, 'CV Folds': cv.n_splits,
                        'ROC AUC': round(auc, 4), 'Std': round(cv_scores.std(), 4)})

    # Load already-saved logistic + svm probs for ensemble
    lr_probs  = pd.read_csv('data/submissions/submission_logistic_regression.csv')['Heart Disease'].values
    svm_probs = pd.read_csv('data/submissions/submission_svm.csv')['Heart Disease'].values

    print("  ── Ensembles ───────────────────────────────────────────")
    ensemble_probs = {
        'ensemble_gb_lr':  np.mean([trained['gradient_boosting_deep'], lr_probs], axis=0),
        'ensemble_gb_rf':  np.mean([trained['gradient_boosting_deep'], trained['random_forest']], axis=0),
        'ensemble_all':    np.mean([trained['gradient_boosting_deep'], trained['gradient_boosting'],
                                    trained['random_forest'], lr_probs, svm_probs], axis=0),
    }
    for ename, probs in ensemble_probs.items():
        out = f'data/submissions/submission_{ename}.csv'
        pd.DataFrame({'id': test_ids, 'Heart Disease': probs}).to_csv(out, index=False)
        print(f"     Saved → {out}")

    results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)
    results_df.to_csv('data/preprocessed/submission_comparison.csv', index=False)

    print(f"\n  ── Model Comparison ────────────────────────────────────")
    print(f"     {'Model':<42} {'Folds':>6} {'ROC AUC':>10} {'Std':>7}")
    print(f"     {'─────':<42} {'─────':>6} {'───────':>10} {'───':>7}")
    for _, row in results_df.iterrows():
        bar = "█" * int(row['ROC AUC'] * 25)
        print(f"     {row['Model']:<42} {int(row['CV Folds']):>6} {row['ROC AUC']:>10.4f} {row['Std']:>7.4f}  {bar}")
    print(f"\n  ✔  Submit: submission_ensemble_all.csv first, then submission_gradient_boosting_deep.csv\n")

if __name__ == "__main__":
    create_submissions()
