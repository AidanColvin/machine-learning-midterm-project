import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._classes import DecisionTreeClassifier as DTC
import matplotlib.patches as mpatches

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv('data/preprocessed/X_train_lasso.csv')
X_test  = pd.read_csv('data/preprocessed/X_test_lasso.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')['Heart Disease']

top6 = ['Thallium', 'Chest pain type', 'Number of vessels fluro',
        'ST depression', 'Max HR', 'Exercise angina']
top6 = [f for f in top6 if f in X_train.columns]

dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1000,
                             criterion='gini', random_state=42)
dt.fit(X_train[top6], y_train)

from sklearn.tree import _tree

def draw_tree(ax, tree, feature_names, node=0,
              x=0.5, y=1.0, dx=0.25, dy=0.18, depth=0):
    """Recursively draw a clean lecture-style tree."""
    left  = tree.children_left[node]
    right = tree.children_right[node]
    is_leaf = left == _tree.TREE_LEAF

    if is_leaf:
        class_idx = np.argmax(tree.value[node][0])
        label = "Yes" if class_idx == 1 else "No"
        ax.text(x, y, label, ha='center', va='center', fontsize=13,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1.5))
    else:
        feat  = feature_names[tree.feature[node]]
        thresh = tree.threshold[node]
        # shorten long feature names
        short = {'Number of vessels fluro': 'Vessels fluro',
                 'Chest pain type': 'Chest pain',
                 'Exercise angina': 'Ex. angina',
                 'ST depression': 'ST depress.',
                 'Max HR': 'Max HR',
                 'Thallium': 'Thallium'}
        label = f"{short.get(feat, feat)} < {thresh:.2f}"
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='square,pad=0.35', fc='white', ec='black', lw=1.5))

        x_left  = x - dx
        x_right = x + dx
        y_child = y - dy

        # draw lines
        ax.plot([x, x_left],  [y - 0.025, y_child + 0.025], 'k-', lw=1.5)
        ax.plot([x, x_right], [y - 0.025, y_child + 0.025], 'k-', lw=1.5)

        # Yes / No branch labels
        ax.text((x + x_left)/2  - 0.01, y - dy/2 + 0.01, 'Yes',
                ha='right', va='center', fontsize=9, color='#333333', style='italic')
        ax.text((x + x_right)/2 + 0.01, y - dy/2 + 0.01, 'No',
                ha='left',  va='center', fontsize=9, color='#333333', style='italic')

        draw_tree(ax, tree, feature_names, left,  x_left,  y_child, dx/2, dy, depth+1)
        draw_tree(ax, tree, feature_names, right, x_right, y_child, dx/2, dy, depth+1)

# ── FIGURE 1: clean single tree ──────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0.55, 1.08)
ax.axis('off')
draw_tree(ax, dt.tree_, top6, dx=0.23, dy=0.15)
ax.set_title('Decision Tree — Heart Disease Classification\n(depth=3, CV-pruned, top 6 features)',
             fontsize=14, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'decision_tree_lecture.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("  ✔ figures/decision_tree_lecture.png")

# ── FIGURE 2: high variance demo — 1 original + 5 bootstrap ──────
from sklearn.utils import resample

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
axes = axes.flatten()

datasets = [(X_train[top6], y_train, 'Original Tree', '#C44E52')]
for i in range(1, 6):
    Xb, yb = resample(X_train[top6], y_train, random_state=i, n_samples=60000)
    datasets.append((Xb, yb, f'b = {i}', '#2d6a2d'))

for ax, (Xb, yb, title, color) in zip(axes, datasets):
    dt_b = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500,
                                   criterion='gini', random_state=42)
    dt_b.fit(Xb, yb)
    ax.set_xlim(0, 1); ax.set_ylim(0.55, 1.08); ax.axis('off')
    draw_tree(ax, dt_b.tree_, top6, dx=0.23, dy=0.15)
    ax.set_title(title, fontsize=12, fontweight='bold', color=color, pad=6)

fig.suptitle('Why Decision Trees Have High Variance\n'
             'Each tree bootstrapped from training data — structure changes substantially',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'decision_tree_variance.png', dpi=130,
            bbox_inches='tight', facecolor='white')
plt.close()
print("  ✔ figures/decision_tree_variance.png")
print("\n  Done.")
