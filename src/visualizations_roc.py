import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

FIGURES_DIR = Path("data/visualizations")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

X_tr_spl = pd.read_csv('data/preprocessed/X_train_spline.csv')
X_te_spl = pd.read_csv('data/preprocessed/X_test_spline.csv')
X_tr_las = pd.read_csv('data/preprocessed/X_train_lasso.csv')
X_te_las = pd.read_csv('data/preprocessed/X_test_lasso.csv')
y_train  = pd.read_csv('data/preprocessed/y_train.csv')['Heart Disease']
y_test   = pd.read_csv('data/preprocessed/y_test.csv')['Heart Disease']

models = {
    'Logistic Regression (Ridge+Spline)': (
        LogisticRegression(solver='lbfgs', C=1.0, max_iter=3000, random_state=42),
        X_tr_spl, X_te_spl, '#4C72B0', '--', 'o'
    ),
    'Random Forest (100 trees)': (
        RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=10,
                               max_features='sqrt', random_state=42, n_jobs=1),
        X_tr_las, X_te_las, '#55A868', '-.', 's'
    ),
    'Gradient Boosting (200 trees)': (
        GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, random_state=42),
        X_tr_las, X_te_las, '#C44E52', '-', '^'
    ),
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── LEFT: zoomed full curve with markers ──────────────────────────
ax = axes[0]
for name, (model, Xtr, Xte, color, ls, marker) in models.items():
    model.fit(Xtr, y_train)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(Xte)[:, 1])
    roc_auc = auc(fpr, tpr)
    # subsample points so markers are visible
    idx = np.unique(np.linspace(0, len(fpr)-1, 40).astype(int))
    ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls, alpha=0.85,
            label=f"{name} (AUC={roc_auc:.4f})")
    ax.plot(fpr[idx], tpr[idx], color=color, marker=marker,
            linestyle='None', markersize=5, alpha=0.7)

ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random (AUC=0.5000)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — Full Range', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.grid(True, alpha=0.3)

# ── RIGHT: zoomed top-left corner where models separate ───────────
ax2 = axes[1]
for name, (model, Xtr, Xte, color, ls, marker) in models.items():
    probs = model.predict_proba(Xte)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    # only plot in zoomed region
    mask = fpr <= 0.25
    idx = np.unique(np.linspace(0, mask.sum()-1, 25).astype(int))
    fpr_z, tpr_z = fpr[mask], tpr[mask]
    ax2.plot(fpr_z, tpr_z, color=color, lw=2.5, linestyle=ls, alpha=0.9,
             label=f"{name} (AUC={roc_auc:.4f})")
    ax2.plot(fpr_z[idx], tpr_z[idx], color=color, marker=marker,
             linestyle='None', markersize=7, alpha=0.9)

ax2.plot([0,0.25],[0,0.25],'k--',lw=1,alpha=0.4,label='Random')
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curves — Zoomed (FPR 0–0.25)\nWhere Models Separate', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=8)
ax2.set_xlim([0, 0.25]); ax2.set_ylim([0.7, 1.01])
ax2.grid(True, alpha=0.3)
# annotate separation zone
ax2.axvspan(0.05, 0.15, alpha=0.06, color='gray', label='_nolegend_')
ax2.text(0.09, 0.72, 'Models\nseparate\nhere', fontsize=8, color='gray', ha='center')

plt.suptitle('ROC Curve Analysis — Heart Disease Classification', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ roc_curves.png saved")
