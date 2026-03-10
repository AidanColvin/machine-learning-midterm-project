import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

FIGURES_DIR     = Path("figures")
SUBMISSIONS_DIR = Path("data/submissions")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_train = pd.read_csv('data/preprocessed/X_train_lasso.csv')
X_test  = pd.read_csv('data/preprocessed/X_test_lasso.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')['Heart Disease']
y_test  = pd.read_csv('data/preprocessed/y_test.csv')['Heart Disease']

# ── FULL MODEL (all 12 lasso features, depth 4) ───────────────────
print("  Training full Decision Tree (12 features, depth=4)...")
dt_full = DecisionTreeClassifier(max_depth=4, min_samples_leaf=500,
                                  criterion='gini', random_state=42)
dt_full.fit(X_train, y_train)

scores = cross_val_score(dt_full, X_train, y_train, cv=CV5, scoring='roc_auc', n_jobs=1)
print(f"  CV AUC : {scores.mean():.4f} ± {scores.std():.4f}")

probs = dt_full.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, probs)
print(f"  Test AUC : {test_auc:.4f}")

# Kaggle submission
test_raw = pd.read_csv('data/raw/test.csv')
ids      = test_raw['id'] if 'id' in test_raw.columns else pd.RangeIndex(len(test_raw))
X_kaggle = test_raw.drop(columns=[c for c in ['id','is_outlier'] if c in test_raw.columns])
X_kaggle = X_kaggle.reindex(columns=X_train.columns, fill_value=0)
kaggle_probs = dt_full.predict_proba(X_kaggle)[:, 1]
pd.DataFrame({'id': ids, 'Heart Disease': kaggle_probs}).to_csv(
    SUBMISSIONS_DIR / 'submission_decision_tree.csv', index=False)
print("  Saved → data/submissions/submission_decision_tree.csv")

# ── VISUAL TREE (top 6 features, depth 4, clean figure) ──────────
print("\n  Generating visual decision tree (top 6 features, depth=4)...")
top6 = ['Thallium', 'Chest pain type', 'Number of vessels fluro',
        'ST depression', 'Max HR', 'Exercise angina']
top6 = [f for f in top6 if f in X_train.columns]

dt_vis = DecisionTreeClassifier(max_depth=4, min_samples_leaf=500,
                                 criterion='gini', random_state=42)
dt_vis.fit(X_train[top6], y_train)

fig, ax = plt.subplots(figsize=(28, 10))
plot_tree(
    dt_vis,
    feature_names=top6,
    class_names=['No Disease', 'Disease'],
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    fontsize=9,
    ax=ax,
    precision=3,
)
ax.set_title('Decision Tree — Top 6 Features (depth=4)\nHeart Disease Classification',
             fontsize=15, fontweight='bold', pad=20)

# color legend
from matplotlib.patches import Patch
legend = [Patch(color='#3B80B3', label='Predicts: Disease'),
          Patch(color='#E8855C', label='Predicts: No Disease')]
ax.legend(handles=legend, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'decision_tree.png', dpi=130, bbox_inches='tight')
plt.close()
print("  ✔ figures/decision_tree.png saved")

print(f"\n  Summary")
print(f"  ─────────────────────────────────────")
print(f"  Full model  : 12 features, depth=4, Test AUC={test_auc:.4f}")
print(f"  Visual tree : top 6 features, depth=4")
print(f"  Submission  : data/submissions/submission_decision_tree.csv")
