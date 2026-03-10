import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.utils import resample

FIGURES_DIR = Path("data/visualizations")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv('data/preprocessed/X_train_lasso.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')['Heart Disease']

top6 = ['Thallium', 'Chest pain type', 'Number of vessels fluro',
        'ST depression', 'Max HR', 'Exercise angina']
top6 = [f for f in top6 if f in X_train.columns]

SHORT = {
    'Number of vessels fluro': 'Vessels fluro',
    'Chest pain type':         'Chest pain type',
    'Exercise angina':         'Exercise angina',
    'ST depression':           'ST depression',
    'Max HR':                  'Max HR',
    'Thallium':                'Thallium',
}

def get_node_positions(tree_, n_features, depth=0, node=0, x=0.5, pos=None, level_width=None):
    if pos is None:
        pos = {}
        level_width = {}
    pos[node] = (x, -depth)
    left  = tree_.children_left[node]
    right = tree_.children_right[node]
    if left != _tree.TREE_LEAF:
        # count leaves to distribute x
        def count_leaves(n):
            l, r = tree_.children_left[n], tree_.children_right[n]
            if l == _tree.TREE_LEAF:
                return 1
            return count_leaves(l) + count_leaves(r)
        ll = count_leaves(left)
        rl = count_leaves(right)
        total = ll + rl
        span = pos[node][0]
        xl = span - (rl / total) * (0.5 ** depth) * 0.9
        xr = span + (ll / total) * (0.5 ** depth) * 0.9
        get_node_positions(tree_, n_features, depth+1, left,  xl, pos, level_width)
        get_node_positions(tree_, n_features, depth+1, right, xr, pos, level_width)
    return pos

def draw_clean_tree(ax, dt, feature_names, title, color='#2d6a2d', fontsize=10):
    tree_ = dt.tree_
    pos   = get_node_positions(tree_, len(feature_names))

    # scale y
    max_depth = max(v[1] for v in pos.values())
    ys = {n: 1.0 + p[1] / abs(max_depth) * 0.85 for n, p in pos.items()}
    xs = {n: p[0] for n, p in pos.items()}

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.08)
    ax.axis('off')

    for node in range(tree_.node_count):
        left  = tree_.children_left[node]
        right = tree_.children_right[node]
        if left == _tree.TREE_LEAF:
            continue
        x0, y0 = xs[node],  ys[node]
        xl, yl  = xs[left],  ys[left]
        xr, yr  = xs[right], ys[right]

        # vertical then horizontal lines like image 3
        mid_y = (y0 + yl) / 2 + 0.01
        # left branch
        ax.plot([x0, x0], [y0 - 0.025, mid_y], color=color, lw=1.8)
        ax.plot([x0, xl], [mid_y, mid_y],        color=color, lw=1.8)
        ax.plot([xl, xl], [mid_y, yl + 0.025],   color=color, lw=1.8)
        # right branch
        ax.plot([x0, xr], [mid_y, mid_y],        color=color, lw=1.8)
        ax.plot([xr, xr], [mid_y, yr + 0.025],   color=color, lw=1.8)

    for node in range(tree_.node_count):
        left  = tree_.children_left[node]
        right = tree_.children_right[node]
        x0, y0 = xs[node], ys[node]
        is_leaf = left == _tree.TREE_LEAF

        if is_leaf:
            class_idx = np.argmax(tree_.value[node][0])
            pct = tree_.value[node][0][class_idx] / tree_.value[node][0].sum() * 100
            label = "Disease" if class_idx == 1 else "No Disease"
            ax.text(x0, y0, f"{label}\n({pct:.0f}%)",
                    ha='center', va='center', fontsize=fontsize-1,
                    fontweight='bold',
                    color='#C44E52' if class_idx == 1 else '#2d6a2d')
        else:
            feat   = feature_names[tree_.feature[node]]
            thresh = tree_.threshold[node]
            label  = f"{SHORT.get(feat, feat)}\n< {thresh:.2f}"
            ax.text(x0, y0, label, ha='center', va='center',
                    fontsize=fontsize, color='black',
                    bbox=dict(boxstyle='square,pad=0.25', fc='white', ec=color, lw=1.4))

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8, color='#222222')

# ── FIGURE 1: Full tree depth 4 all 6 features ───────────────────
print("  Full tree (depth 4)...")
dt_full = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2000,
                                  criterion='gini', random_state=42)
dt_full.fit(X_train[top6], y_train)
fig, ax = plt.subplots(figsize=(20, 10))
draw_clean_tree(ax, dt_full, top6,
    title='Full Decision Tree — Heart Disease Classification\n'
          'Top 6 Clinical Features  |  depth = 4  |  Gini criterion',
    color='#2d6a2d', fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'decision_tree_full.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("  ✔ decision_tree_full.png")

# ── FIGURE 2: Simple 3-feature depth-2 for non-technical reader ───
print("  Simple non-technical tree (depth 2, 3 features)...")
top3 = ['Thallium', 'Chest pain type', 'Number of vessels fluro']
dt_simple = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5000,
                                    criterion='gini', random_state=42)
dt_simple.fit(X_train[top3], y_train)
fig, ax = plt.subplots(figsize=(12, 7))
draw_clean_tree(ax, dt_simple, top3,
    title='Simple Decision Tree — For Non-Technical Readers\n'
          '3 Key Features: Thallium scan, Chest pain type, Vessels blocked',
    color='#2d6a2d', fontsize=11)
# add plain-english annotations
ax.text(0.5, 0.03,
    'How to read: Start at the top. Follow Yes (left) or No (right) at each split.\n'
    'Leaf nodes show the predicted outcome and confidence.',
    ha='center', va='bottom', fontsize=9, style='italic', color='#555555',
    transform=ax.transAxes)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'decision_tree_simple.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("  ✔ decision_tree_simple.png")

# ── FIGURE 3: High variance demo 1+5 bootstrap ───────────────────
print("  High variance demo...")
fig, axes = plt.subplots(2, 3, figsize=(21, 12))
axes = axes.flatten()
datasets = [(X_train[top3], y_train, 'Original Tree', '#C44E52')]
for i in range(1, 6):
    Xb, yb = resample(X_train[top3], y_train, random_state=i, n_samples=60000)
    datasets.append((Xb, yb, f'Bootstrap  b = {i}', '#2d6a2d'))
for ax, (Xb, yb, title, color) in zip(axes, datasets):
    dt_b = DecisionTreeClassifier(max_depth=2, min_samples_leaf=500,
                                   criterion='gini', random_state=42)
    dt_b.fit(Xb, yb)
    draw_clean_tree(ax, dt_b, top3, title=title, color=color, fontsize=10)
fig.suptitle(
    'Decision Trees Have High Variance\n'
    'A small change in training data produces a different tree structure',
    fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'decision_tree_variance.png', dpi=130,
            bbox_inches='tight', facecolor='white')
plt.close()
print("  ✔ decision_tree_variance.png")
print("\n  All 3 figures saved to figures/")
