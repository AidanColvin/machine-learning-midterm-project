import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

CV5  = StratifiedKFold(n_splits=5,  shuffle=True, random_state=42)
CV10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

FEATURE_COLS = [
    'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
    'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
    'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

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

def scale_test_like_train(X_train_raw, X_test_raw):
    scaler         = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test_raw),      columns=X_test_raw.columns)
    return X_train_scaled, X_test_scaled

def lasso_select(X_train, y_train, X_test):
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_train, y_train.values)
    mask  = np.abs(lasso.coef_) > 0
    cols  = X_train.columns[mask].tolist()
    if len(cols) < 3:
        cols = X_train.columns.tolist()
    print(f"     Lasso kept {len(cols)}/{X_train.shape[1]} features: {cols}")
    return X_train[cols], X_test[cols], cols

def add_spline_features(X_train, X_test, continuous_cols, n_knots=5, degree=3):
    spl      = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
    tr_spl   = spl.fit_transform(X_train[continuous_cols])
    te_spl   = spl.transform(X_test[continuous_cols])
    n_out    = tr_spl.shape[1]
    per_feat = n_out // len(continuous_cols)
    names    = [f"spline_{c}_{i}" for c in continuous_cols for i in range(per_feat)]
    names    = names[:n_out]
    tr_df = pd.DataFrame(tr_spl, columns=names, index=X_train.index)
    te_df = pd.DataFrame(te_spl, columns=names, index=X_test.index)
    return pd.concat([X_train, tr_df], axis=1), pd.concat([X_test, te_df], axis=1)

def tune_C(X, y, cv):
    param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    gs = GridSearchCV(
        LogisticRegression(max_iter=3000, random_state=42, solver='lbfgs'),
        param_grid, cv=cv, scoring='roc_auc', n_jobs=1
    )
    gs.fit(X, y.values)
    print(f"     Best C={gs.best_params_['C']}  (CV AUC={gs.best_score_:.4f})")
    return gs.best_params_['C']

def create_submissions() -> None:
    # ── Load raw data ────────────────────────────────────────────────
    raw_train = pd.read_csv('data/raw/train.csv')
    raw_test  = pd.read_csv('data/raw/test.csv')

    # Encode target: Presence=1, Absence=0 (handles both string and int)
    y = raw_train['Heart Disease']
    if y.dtype == object or pd.api.types.is_string_dtype(y):
        y = y.map({'Absence': 0, 'Presence': 1})
    y = y.astype(int)

    X_raw      = raw_train.drop(columns=[c for c in ['Heart Disease','id','is_outlier'] if c in raw_train.columns])
    X_raw      = X_raw.reindex(columns=FEATURE_COLS, fill_value=0)

    test_ids   = raw_test['id']
    X_test_raw = raw_test.drop(columns=[c for c in ['id','is_outlier'] if c in raw_test.columns])
    X_test_raw = X_test_raw.reindex(columns=FEATURE_COLS, fill_value=0)

    print_dataset_stats(X_raw, y, label="Raw Training")

    # ── Scale: fit on train only, apply to both ──────────────────────
    print("  ── Scaling (fit on train → apply to train & test) ──────")
    X_scaled, X_test_scaled = scale_test_like_train(X_raw, X_test_raw)
    print(f"     Train: {X_scaled.shape}  |  Test: {X_test_scaled.shape}\n")
    print_dataset_stats(X_scaled, y, label="Scaled Training")

    # ── Lasso feature selection ──────────────────────────────────────
    print("  ── Feature Selection (LassoCV, cv=5) ───────────────────")
    X_sel, X_test_sel, sel_cols = lasso_select(X_scaled, y, X_test_scaled)

    # ── Spline features on continuous cols ──────────────────────────
    continuous  = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    cont_in_sel = [c for c in continuous if c in sel_cols]
    print(f"\n  ── Spline Features (n_knots=5, degree=3) ───────────────")
    X_spline, X_test_spline = add_spline_features(X_sel, X_test_sel, cont_in_sel)
    print(f"     Features after splines: {X_spline.shape[1]}\n")

    # ── Tune C (10-fold GridSearch) ──────────────────────────────────
    print("  ── C Tuning via 10-fold GridSearchCV ───────────────────")
    best_C = tune_C(X_spline, y, CV10)

    os.makedirs('data/submissions', exist_ok=True)

    models = {
        'logistic_ridge_spline': (
            'Logistic Ridge + Spline (10-fold)',
            LogisticRegression(C=best_C, max_iter=3000, random_state=42, solver='lbfgs'),
            X_spline, X_test_spline, CV10
        ),
        'logistic_lasso_spline': (
            'Logistic L1 + Spline (10-fold)',
            LogisticRegression(C=best_C, penalty='l1', max_iter=3000, random_state=42, solver='liblinear'),
            X_spline, X_test_spline, CV10
        ),
        'gradient_boosting': (
            'Gradient Boosting (5-fold)',
            GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=42
            ),
            X_sel, X_test_sel, CV5
        ),
        'random_forest': (
            'Random Forest (5-fold)',
            RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=10,
                max_features='sqrt', random_state=42, n_jobs=1
            ),
            X_sel, X_test_sel, CV5
        ),
        'svm_spline': (
            'SVM + Spline (5-fold)',
            CalibratedClassifierCV(LinearSVC(C=0.1, random_state=42, dual=False, max_iter=3000), cv=3),
            X_spline, X_test_spline, CV5
        ),
    }

    results = []
    for name, (label, model, X_tr, X_te, cv) in models.items():
        pad = '─' * max(1, 50 - len(label))
        print(f"  ── {label} {pad}")
        cv_scores = cross_val_score(model, X_tr, y, cv=cv, scoring='roc_auc', n_jobs=1)
        auc = cv_scores.mean()
        print(f"     CV AUC : {auc:.4f}  std={cv_scores.std():.4f}  folds={cv.n_splits}")
        print(f"     Training on 100% of data...")
        model.fit(X_tr, y)
        probs = model.predict_proba(X_te)[:, 1]
        out   = f'data/submissions/submission_{name}.csv'
        pd.DataFrame({'id': test_ids, 'Heart Disease': probs}).to_csv(out, index=False)
        print(f"     Saved → {out}\n")
        results.append({'Model': label, 'CV Folds': cv.n_splits, 'ROC AUC': round(auc, 4), 'Std': round(cv_scores.std(), 4)})

    results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)
    results_df.to_csv('data/preprocessed/submission_comparison.csv', index=False)

    print("  ── Final Model Comparison ──────────────────────────────")
    print(f"     {'Model':<40} {'Folds':>6} {'ROC AUC':>10} {'Std':>7}")
    print(f"     {'─────':<40} {'─────':>6} {'───────':>10} {'───':>7}")
    for _, row in results_df.iterrows():
        bar = "█" * int(row['ROC AUC'] * 25)
        print(f"     {row['Model']:<40} {int(row['CV Folds']):>6} {row['ROC AUC']:>10.4f} {row['Std']:>7.4f}  {bar}")

    best = results_df.iloc[0]
    print(f"\n  ✔  Best: {best['Model']}  ({best['ROC AUC']:.4f} ROC AUC)")
    print(f"  ── Saved → data/preprocessed/submission_comparison.csv\n")

if __name__ == "__main__":
    create_submissions()
