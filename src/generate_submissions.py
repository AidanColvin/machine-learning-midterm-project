import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

CV5  = StratifiedKFold(n_splits=5,  shuffle=True, random_state=42)
CV10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

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
    # Use preprocessed (already scaled) data — this is what scored 0.95 on Kaggle
    train_df = pd.read_csv('data/preprocessed/preprocessed-train-data.csv')
    X = train_df.drop(columns=[c for c in ['Heart Disease','id','is_outlier'] if c in train_df.columns])
    y = train_df['Heart Disease']
    if pd.api.types.is_string_dtype(y):
        y = y.map({'Absence': 0, 'Presence': 1}).astype(int)

    test_df  = pd.read_csv('data/raw/test.csv')
    test_ids = test_df['id']
    X_test   = test_df.drop(columns=[c for c in ['id','is_outlier'] if c in test_df.columns])

    # Scale test using train stats (preprocessed train is already scaled, test is raw)
    scaler = StandardScaler()
    scaler.fit(X)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test.reindex(columns=X.columns, fill_value=0)), columns=X.columns)

    print_dataset_stats(X, y, label="Preprocessed Training")
    print_dataset_stats(X_test_scaled, pd.Series([0]*len(X_test_scaled)), label="Scaled Test")

    os.makedirs('data/submissions', exist_ok=True)

    models = {
        'gb_deep': (
            'Gradient Boosting Deep (5-fold)',
            GradientBoostingClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.03,
                subsample=0.8, min_samples_leaf=10,
                max_features=0.8, random_state=42
            ),
            CV5
        ),
        'gb_wide': (
            'Gradient Boosting Wide (5-fold)',
            GradientBoostingClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.7, min_samples_leaf=5,
                max_features='sqrt', random_state=42
            ),
            CV5
        ),
        'rf_deep': (
            'Random Forest Deep (10-fold)',
            RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_leaf=5,
                max_features='sqrt', random_state=42, n_jobs=1
            ),
            CV10
        ),
        'logistic_high_c': (
            'Logistic Regression C=10 (10-fold)',
            LogisticRegression(C=10, max_iter=3000, solver='lbfgs', random_state=42),
            CV10
        ),
    }

    results = []
    for name, (label, model, cv) in models.items():
        pad = '─' * max(1, 52 - len(label))
        print(f"  ── {label} {pad}")
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        auc = cv_scores.mean()
        print(f"     CV AUC : {auc:.4f}  std={cv_scores.std():.4f}  folds={cv.n_splits}")
        print(f"     Training on 100% of data...")
        model.fit(X, y)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        out   = f'data/submissions/submission_{name}.csv'
        pd.DataFrame({'id': test_ids, 'Heart Disease': probs}).to_csv(out, index=False)
        print(f"     Saved → {out}\n")
        results.append({'Model': label, 'CV Folds': cv.n_splits, 'ROC AUC': round(auc, 4), 'Std': round(cv_scores.std(), 4)})

    # Ensemble: average probabilities from gb_deep + rf_deep
    print("  ── Ensemble: GB Deep + RF Deep (avg probs) ────────────")
    gb  = [m for n,(l,m,c) in models.items() if n == 'gb_deep'][0]
    rf  = [m for n,(l,m,c) in models.items() if n == 'rf_deep'][0]
    ensemble_probs = (gb.predict_proba(X_test_scaled)[:,1] + rf.predict_proba(X_test_scaled)[:,1]) / 2
    pd.DataFrame({'id': test_ids, 'Heart Disease': ensemble_probs}).to_csv('data/submissions/submission_ensemble.csv', index=False)
    print(f"     Saved → data/submissions/submission_ensemble.csv\n")

    results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)
    results_df.to_csv('data/preprocessed/submission_comparison.csv', index=False)

    print("  ── Final Model Comparison ──────────────────────────────")
    print(f"     {'Model':<42} {'Folds':>6} {'ROC AUC':>10} {'Std':>7}")
    print(f"     {'─────':<42} {'─────':>6} {'───────':>10} {'───':>7}")
    for _, row in results_df.iterrows():
        bar = "█" * int(row['ROC AUC'] * 25)
        print(f"     {row['Model']:<42} {int(row['CV Folds']):>6} {row['ROC AUC']:>10.4f} {row['Std']:>7.4f}  {bar}")

    best = results_df.iloc[0]
    print(f"\n  ✔  Best: {best['Model']}  ({best['ROC AUC']:.4f} ROC AUC)")
    print(f"  ── Also try: submission_ensemble.csv")
    print(f"  ── Saved → data/preprocessed/submission_comparison.csv\n")

if __name__ == "__main__":
    create_submissions()
