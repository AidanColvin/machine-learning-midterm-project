import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression, LassoCV, RidgeCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, StratifiedKFold

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

def lasso_feature_selection(X, y):
    """Use LassoCV to select most predictive features."""
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X, y)
    mask = np.abs(lasso.coef_) > 0
    selected = X.columns[mask].tolist()
    if len(selected) < 3:
        selected = X.columns.tolist()
    print(f"     Lasso selected {len(selected)}/{X.shape[1]} features: {selected}")
    return selected

def create_submissions() -> None:
    train_df = pd.read_csv('data/preprocessed/preprocessed-train-data.csv')
    X_train  = train_df.drop(['Heart Disease', 'id', 'is_outlier'], axis=1, errors='ignore')
    y_train  = train_df['Heart Disease']

    test_df  = pd.read_csv('data/raw/test.csv')
    test_ids = test_df['id']
    X_test   = test_df.drop(['id', 'is_outlier'], axis=1, errors='ignore')
    X_test   = X_test.reindex(columns=X_train.columns, fill_value=0)

    print_dataset_stats(X_train, y_train)

    # --- Lasso feature selection ---
    print("  ── Feature Selection (Lasso) ───────────────────")
    selected_features = lasso_feature_selection(X_train, y_train)
    X_train_sel = X_train[selected_features]
    X_test_sel  = X_test[selected_features]

    # --- Polynomial features on selected (degree=2, no bias) ---
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_sel)
    X_test_poly  = poly.transform(X_test_sel)
    print(f"     Polynomial features expanded to: {X_train_poly.shape[1]} features\n")

    os.makedirs('data/submissions', exist_ok=True)

    models = {
        'logistic_regression_lasso': (
            'Logistic Regression + Lasso Features',
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(C=1.0, max_iter=2000, random_state=42))
            ]),
            X_train_sel, X_test_sel
        ),
        'logistic_regression_poly': (
            'Logistic Regression + Poly Features',
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(C=0.1, max_iter=2000, random_state=42))
            ]),
            X_train_poly, X_test_poly
        ),
        'ridge_logistic': (
            'Ridge Logistic Regression',
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(penalty='l2', C=0.01, max_iter=2000, solver='lbfgs', random_state=42))
            ]),
            X_train_sel, X_test_sel
        ),
        'gradient_boosting': (
            'Gradient Boosting',
            GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=42
            ),
            X_train_sel, X_test_sel
        ),
        'random_forest': (
            'Random Forest + Feature Selection',
            RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=10,
                max_features='sqrt', random_state=42, n_jobs=1
            ),
            X_train_sel, X_test_sel
        ),
    }

    results = []

    for name, (label, model, X_tr, X_te) in models.items():
        print(f"  ── {label} ──────────────────────────────")
        cv_scores = cross_val_score(model, X_tr, y_train, cv=CV, scoring='roc_auc', n_jobs=1)
        auc = cv_scores.mean()
        print(f"     CV ROC AUC : {auc:.4f}  (std={cv_scores.std():.4f})")
        print(f"     Training on full dataset...")
        model.fit(X_tr, y_train)
        probabilities = model.predict_proba(X_te)[:, 1]
        out = f'data/submissions/submission_{name}.csv'
        pd.DataFrame({'id': test_ids, 'Heart Disease': probabilities}).to_csv(out, index=False)
        print(f"     Saved → {out}\n")
        results.append({'Model': label, 'ROC AUC': round(auc, 4)})

    results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)
    results_df.to_csv('data/preprocessed/submission_comparison.csv', index=False)

    print("  ── Model Comparison ────────────────────────────")
    print(f"     {'Model':<40} {'ROC AUC':>10}")
    print(f"     {'─────':<40} {'───────':>10}")
    for _, row in results_df.iterrows():
        bar = "█" * int(row['ROC AUC'] * 30)
        print(f"     {row['Model']:<40} {row['ROC AUC']:>10.4f}  {bar}")

    best = results_df.iloc[0]
    bfile = best['Model'].lower().replace(' ', '_').replace('+', '').replace('  ', '_')
    print(f"\n  ✔  Best: {best['Model']}  ({best['ROC AUC']:.4f} ROC AUC)")
    print(f"  ── Saved → data/preprocessed/submission_comparison.csv\n")

if __name__ == "__main__":
    create_submissions()
