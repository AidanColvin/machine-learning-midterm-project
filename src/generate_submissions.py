import pandas as pd
import os
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def print_dataset_stats(X_train, y_train) -> None:
    print("\n  ▶  Dataset Summary")
    print("  ══════════════════════════════════════════════")
    print(f"     Sample size       : {len(X_train):,} rows")
    print(f"     Number of features: {X_train.shape[1]}")
    print(f"     Dependent variable: Heart Disease (binary)")
    counts = y_train.value_counts()
    for label, count in counts.items():
        bar = "█" * int((count / len(y_train)) * 20)
        print(f"     {str(label):<12} {bar}  {count:,} ({count/len(y_train)*100:.1f}%)")
    print("\n  ── Feature Descriptive Stats ───────────────────")
    print(f"     {'Feature':<35} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"     {'───────':<35} {'────':>7} {'───':>7} {'───':>7} {'───':>7}")
    for col in X_train.columns:
        s = X_train[col]
        print(f"     {col:<35} {s.mean():>7.2f} {s.std():>7.2f} {s.min():>7.2f} {s.max():>7.2f}")
    print("  ══════════════════════════════════════════════\n")

def create_submissions() -> None:
    train_df = pd.read_csv('data/preprocessed/preprocessed-train-data.csv')
    X_train  = train_df.drop(['Heart Disease', 'id', 'is_outlier'], axis=1, errors='ignore')
    y_train  = train_df['Heart Disease']

    test_df  = pd.read_csv('data/raw/test.csv')
    test_ids = test_df['id']
    X_test   = test_df.drop(['id', 'is_outlier'], axis=1, errors='ignore')
    X_test   = X_test.reindex(columns=X_train.columns, fill_value=0)

    print_dataset_stats(X_train, y_train)

    os.makedirs('data/submissions', exist_ok=True)

    models = {
        'logistic_regression' : LogisticRegression(max_iter=2000),
        'svm'                 : CalibratedClassifierCV(LinearSVC(random_state=42, dual=False), cv=3),
        'random_forest'       : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        'gradient_boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = []

    for name, model in models.items():
        label = name.replace('_', ' ').title()
        print(f"  ── {label} ──────────────────────────────")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=1)
        auc = cv_scores.mean()
        print(f"     Cross-Validated ROC AUC : {auc:.4f}")
        print(f"     Training on full dataset...")
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        out = f'data/submissions/submission_{name}.csv'
        pd.DataFrame({'id': test_ids, 'Heart Disease': probabilities}).to_csv(out, index=False)
        print(f"     Saved → {out}\n")
        results.append({'Model': label, 'ROC AUC': round(auc, 4)})

    results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)
    results_df.to_csv('data/preprocessed/submission_comparison.csv', index=False)

    print("  ── Model Comparison ────────────────────────────")
    print(f"     {'Model':<30} {'ROC AUC':>10}")
    print(f"     {'─────':<30} {'───────':>10}")
    for _, row in results_df.iterrows():
        bar = "█" * int(row['ROC AUC'] * 30)
        print(f"     {row['Model']:<30} {row['ROC AUC']:>10.4f}  {bar}")

    best = results_df.iloc[0]
    bfile = best['Model'].lower().replace(' ', '_')
    print(f"\n  ✔  Best: submission_{bfile}.csv  ({best['ROC AUC']:.4f} ROC AUC)")
    print(f"  ── Saved → data/preprocessed/submission_comparison.csv\n")

if __name__ == "__main__":
    create_submissions()
