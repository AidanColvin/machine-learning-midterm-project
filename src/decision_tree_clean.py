import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import tree

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_train = pd.read_csv('data/preprocessed/X_train_lasso.csv')
X_test  = pd.read_csv('data/preprocessed/X_test_lasso.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')['Heart Disease']
y_test  = pd.read_csv('data/preprocessed/y_test.csv')['Heart Disease']

top6 = ['Thallium', 'Chest pain type', 'Number of vessels fluro',
        'ST depression', 'Max HR', 'Exercise angina']
top6 = [f for f in top6 if f in X_train.columns]

# ── CV to find optimal depth ───────────────────────────────────────
print("  Finding optimal depth via 5-fold CV...")
print(f"  {'Depth':<8} {'CV AUC':>10} {'Std':>8}")
print(f"  {'─────':<8} {'──────':>10} {'───':>8}")
best_depth, best_score = 3, 0
for depth in range(2, 8):
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=200,
                                 criterion='gini', random_state=42)
    scores = cross_val_score(dt, X_train[top6], y_train, cv=CV5,
                             scoring='roc_auc', n_jobs=1)
    marker = " ◄ best" if scores.mean() > best_score else ""
    print(f"  {depth:<8} {scores.mean():>10.4f} {scores.std():>8.4f}{marker}")
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_depth = depth

print(f"\n  Optimal depth: {best_depth}  (CV AUC={best_score:.4f})")

# ── FIT FINAL TREE at optimal depth ───────────────────────────────
dt_final = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=200,
                                   criterion='gini', random_state=42)
dt_final.fit(X_train[top6], y_train)
test_auc = roc_auc_score(y_test, dt_final.predict_proba(X_test[top6])[:, 1])
print(f"  Test AUC: {test_auc:.4f}")

# ── PRINT TEXT TREE (like lecture slides) ─────────────────────────
print("\n  Text representation:")
print(export_text(dt_final, feature_names=top6))

# ── FIGURE 1: CLEAN READABLE TREE (lecture style) ─────────────────
print("  Plotting clean readable tree...")
fig, ax = plt.subplots(figsize=(22, 9))
tree.plot_tree(
    dt_final,
    feature_names=top6,
    class_names=['No Disease', 'Disease'],
    filled=False,          # white background like lecture slides
    rounded=False,         # square boxes like lecture style
    impurity=True,
    proportion=True,       # show proportions not raw counts
    fontsize=10,
    ax=ax,
    precision=2,
)
ax.set_title(
    f'Decision Tree — Top 6 Features (depth={best_depth}, CV-selected)\n'
    f'CV AUC = {best_score:.4f}  |  Test AUC = {test_auc:.4f}',
    fontsize=13, fontweight='bold', pad=15
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'decision_tree_clean.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  ✔ figures/decision_tree_clean.png")

# ── FIGURE 2: HIGH VARIANCE DEMO (like lecture slide 3) ───────────
print("  Plotting high variance demo (5 bootstrap trees)...")
from sklearn.utils import resample

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Original tree
ax0 = axes[0]
dt0 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200,
                               criterion='gini', random_state=42)
dt0.fit(X_train[top6], y_train)
tree.plot_tree(dt0, feature_names=top6, class_names=['No','Yes'],
               filled=False, rounded=False, impurity=False,
               proportion=True, fontsize=8, ax=ax0, precision=2)
ax0.set_title('Original Tree', fontsize=11, fontweight='bold', color='#C44E52')

# 5 bootstrap trees
for i in range(1, 6):
    X_boot, y_boot = resample(X_train[top6], y_train, random_state=i, n_samples=50000)
    dt_b = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200,
                                   criterion='gini', random_state=i)
    dt_b.fit(X_boot, y_boot)
    tree.plot_tree(dt_b, feature_names=top6, class_names=['No','Yes'],
                   filled=False, rounded=False, impurity=False,
                   proportion=True, fontsize=8, ax=axes[i], precision=2)
    axes[i].set_title(f'b = {i}', fontsize=11, fontweight='bold', color='#55A868')

fig.suptitle(
    'Decision Trees Have High Variance\n'
    'Each tree bootstrapped from training data — structure changes substantially',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'decision_tree_variance.png', dpi=130, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  ✔ figures/decision_tree_variance.png")

print("\n  Done. Saved:")
print("     figures/decision_tree_clean.png   — lecture-style readable tree")
print("     figures/decision_tree_variance.png — high variance bootstrap demo")
