import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_train_spl = pd.read_csv('data/preprocessed/X_train_spline.csv')
X_test_spl  = pd.read_csv('data/preprocessed/X_test_spline.csv')
X_train_las = pd.read_csv('data/preprocessed/X_train_lasso.csv')
X_test_las  = pd.read_csv('data/preprocessed/X_test_lasso.csv')
y_train     = pd.read_csv('data/preprocessed/y_train.csv')['Heart Disease']
y_test      = pd.read_csv('data/preprocessed/y_test.csv')['Heart Disease']

models = {
    'Logistic Regression Ridge Spline': (
        LogisticRegression(C=1.0, solver='lbfgs', max_iter=3000, random_state=42),
        X_train_spl, X_test_spl
    ),
    'Random Forest 100 trees': (
        RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=10,
                               max_features='sqrt', random_state=42, n_jobs=1),
        X_train_las, X_test_las
    ),
    'Gradient Boosting 200 trees': (
        GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, random_state=42),
        X_train_las, X_test_las
    ),
}

print("\n  RESULTS AND EVALUATION")
print("  ══════════════════════════════════════════════════════════")

print("\n  5-Fold CV AUC")
print(f"  {'Model':<40} {'AUC':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print(f"  {'─────':<40} {'───':>8} {'───':>8} {'───':>8} {'───':>8}")
for name, (model, Xtr, Xte) in models.items():
    scores = cross_val_score(model, Xtr, y_train, cv=CV5, scoring='roc_auc', n_jobs=1)
    print(f"  {name:<40} {scores.mean():>8.4f} {scores.std():>8.4f} {scores.min():>8.4f} {scores.max():>8.4f}  {'█'*int(scores.mean()*30)}")

print("\n  Test Set Metrics")
print(f"  {'Model':<40} {'AUC':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
print(f"  {'─────':<40} {'───':>7} {'───':>7} {'────':>7} {'──────':>7} {'──':>7}")

test_results = {}
trained = {}
for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    probs = model.predict_proba(Xte)[:, 1]
    preds = (probs >= 0.5).astype(int)
    res = {
        'AUC':       roc_auc_score(y_test, probs),
        'Accuracy':  accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall':    recall_score(y_test, preds),
        'F1':        f1_score(y_test, preds),
    }
    test_results[name] = res
    trained[name] = (model, probs, Xte)
    print(f"  {name:<40} {res['AUC']:>7.4f} {res['Accuracy']:>7.4f} {res['Precision']:>7.4f} {res['Recall']:>7.4f} {res['F1']:>7.4f}")

print("\n  Confusion Matrices")
for name, (model, probs, Xte) in trained.items():
    cm = confusion_matrix(y_test, (probs >= 0.5).astype(int))
    print(f"\n  {name}")
    print(f"                 Predicted 0   Predicted 1")
    print(f"  Actual 0       {cm[0,0]:>10,}  {cm[0,1]:>10,}")
    print(f"  Actual 1       {cm[1,0]:>10,}  {cm[1,1]:>10,}")

print("\n  Feature Importance - Gradient Boosting Top 10")
gb = [m for n,(m,p,x) in trained.items() if 'Gradient' in n][0]
imps = pd.Series(gb.feature_importances_, index=X_train_las.columns).sort_values(ascending=False).head(10)
for feat, imp in imps.items():
    print(f"  {feat:<28} {imp:.4f}  {'█'*int(imp*100)}")

baseline = test_results['Logistic Regression Ridge Spline']['AUC']
print("\n  Comparative AUC vs Logistic Baseline")
for name, res in test_results.items():
    diff = res['AUC'] - baseline
    sign = "+" if diff >= 0 else ""
    print(f"  {name:<40} {res['AUC']:.4f}  {sign}{diff:.4f}")

pd.DataFrame(test_results).T.reset_index().rename(columns={'index': 'Model'}).to_csv(
    'data/preprocessed/model_comparison.csv', index=False)
print("\n  Saved to data/preprocessed/model_comparison.csv")
print("  ══════════════════════════════════════════════════════════\n")
