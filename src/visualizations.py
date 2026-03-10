import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix

FIGURES_DIR = Path("data/visualizations")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CV5     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
PALETTE = ['#4C72B0', '#55A868', '#C44E52']

X_tr_spl = pd.read_csv('data/preprocessed/X_train_spline.csv')
X_te_spl = pd.read_csv('data/preprocessed/X_test_spline.csv')
X_tr_las = pd.read_csv('data/preprocessed/X_train_lasso.csv')
X_te_las = pd.read_csv('data/preprocessed/X_test_lasso.csv')
y_train  = pd.read_csv('data/preprocessed/y_train.csv')['Heart Disease']
y_test   = pd.read_csv('data/preprocessed/y_test.csv')['Heart Disease']

models = {
    'Logistic Regression (Ridge+Spline)': (
        LogisticRegression(solver='lbfgs', C=1.0, max_iter=3000, random_state=42),
        X_tr_spl, X_te_spl
    ),
    'Random Forest (100 trees)': (
        RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=10,
                               max_features='sqrt', random_state=42, n_jobs=1),
        X_tr_las, X_te_las
    ),
    'Gradient Boosting (200 trees)': (
        GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, random_state=42),
        X_tr_las, X_te_las
    ),
}

print("  Training models...")
trained = {}
for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    probs = model.predict_proba(Xte)[:, 1]
    preds = (probs >= 0.5).astype(int)
    trained[name] = (model, probs, preds, Xtr, Xte)
    print(f"  ✔ {name}")

# ── 1. ROC CURVES ─────────────────────────────────────────────────
print("\n  ROC curves...")
fig, ax = plt.subplots(figsize=(7, 6))
for (name, (model, probs, preds, Xtr, Xte)), color in zip(trained.items(), PALETTE):
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc(fpr,tpr):.4f})")
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random (AUC = 0.5000)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ roc_curves.png")

# ── 2. FEATURE IMPORTANCE ─────────────────────────────────────────
print("  Feature importance...")
gb = trained['Gradient Boosting (200 trees)'][0]
imps = pd.Series(gb.feature_importances_, index=X_tr_las.columns).sort_values(ascending=True).tail(12)
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(imps.index, imps.values, color='#4C72B0', edgecolor='white', height=0.7)
for bar, val in zip(bars, imps.values):
    ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
ax.set_title('Feature Importance — Gradient Boosting', fontsize=14, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ feature_importance.png")

# ── 3. CONFUSION MATRICES ─────────────────────────────────────────
print("  Confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, (model, probs, preds, Xtr, Xte)), color in zip(axes, trained.items(), PALETTE):
    cm = confusion_matrix(y_test, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', ax=ax, cmap='Blues', cbar=False,
                xticklabels=['No Disease','Disease'], yticklabels=['No Disease','Disease'])
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.75, f'n={cm[i,j]:,}', ha='center', va='center', fontsize=7, color='gray')
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
fig.suptitle('Confusion Matrices (% of Actual Class)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ confusion_matrices.png")

# ── 4. CV AUC COMPARISON ──────────────────────────────────────────
print("  CV AUC comparison...")
cv_means, cv_stds = [], []
for name, (model, probs, preds, Xtr, Xte) in trained.items():
    scores = cross_val_score(model, Xtr, y_train, cv=CV5, scoring='roc_auc', n_jobs=1)
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())
    print(f"    {name} = {scores.mean():.4f} ± {scores.std():.4f}")

short = ['Logistic\n(Ridge+Spline)', 'Random\nForest', 'Gradient\nBoosting']
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(short))
bars = ax.bar(x, cv_means, yerr=cv_stds, capsize=6, color=PALETTE,
              edgecolor='white', width=0.5, error_kw={'elinewidth':2,'ecolor':'black'})
for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x()+bar.get_width()/2, mean+std+0.0005,
            f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=11)
ax.set_ylabel('ROC AUC', fontsize=12)
ax.set_title('5-Fold Cross-Validation AUC Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([0.94, 0.962])
ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='0.95 reference')
ax.grid(True, axis='y', alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'cv_auc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ cv_auc_comparison.png")

print(f"\n  ✔  All 4 figures saved to figures/\n")
