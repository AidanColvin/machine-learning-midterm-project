import pandas as pd
import numpy as np
import sklearn
import sys

print("""
  METHODOLOGY REPORT
  ══════════════════════════════════════════════════════════════════════

  ── OVERVIEW OF APPROACHES ─────────────────────────────────────────

  Three model families were trained and evaluated:

  1. Logistic Regression (Ridge regularization + Spline features)
     - Linear model with L2 penalty to prevent overfitting
     - Spline transformation applied to 5 continuous features
       (Age, BP, Cholesterol, Max HR, ST depression) to capture
       non-linear relationships within a linear framework
     - Hyperparameter C tuned via GridSearchCV over
       [0.001, 0.01, 0.1, 1, 10, 100]
     - Serves as interpretable baseline

  2. Random Forest (100 trees)
     - Ensemble of decision trees using bagging + feature subsampling
     - Captures non-linear interactions without feature engineering
     - Hyperparameters: max_depth=15, min_samples_leaf=10,
       max_features=sqrt(p), n_estimators=100
     - Limited to 100 trees due to Codespace memory constraints

  3. Gradient Boosting (200 trees)
     - Sequential ensemble: each tree corrects errors of the prior
     - Stronger than Random Forest on structured tabular data
     - Hyperparameters: n_estimators=200, max_depth=4,
       learning_rate=0.05, subsample=0.8
     - Selected as final submission model based on highest CV AUC

  ── RATIONALE FOR CHOSEN METHOD ────────────────────────────────────

  Gradient Boosting was selected as the final model for four reasons:

  (1) Highest CV AUC: 0.9543 vs RF 0.9530 vs LR 0.9510
  (2) Handles mixed feature types natively (binary, ordinal,
      continuous) without requiring spline preprocessing
  (3) Robust to outliers via shallow trees (max_depth=4) and
      subsampling (subsample=0.8)
  (4) learning_rate=0.05 with 200 trees provides better
      bias-variance tradeoff than fewer deeper trees

  Logistic Regression was retained as baseline. Despite lower AUC,
  its coefficients are interpretable and provide clinical insight.
  Random Forest was competitive but marginally below GB on all metrics.

  Computational constraint: Codespace RAM (~8GB) required limiting
  RF to 100 trees and running GB without parallelism (n_jobs=1).

  ── IMPLEMENTATION DETAILS ─────────────────────────────────────────

  Data Preprocessing (src/preprocessing.py):
    - Raw data: 630,000 rows x 13 features, no missing values
    - Target encoded: Presence=1, Absence=0
    - Continuous features z-score standardized (fit on train only)
    - Outliers flagged at |z| > 3.0 via is_outlier column (retained)

  Feature Engineering (src/feature_engineering.py):
    - LassoCV (5-fold) fit on training data to select features
    - Dropped: FBS over 120 (coefficient shrunk to zero)
    - Retained: 12/13 features
    - SplineTransformer (n_knots=5, degree=3) applied to continuous
      features for Logistic Regression input (42 total features)
    - Tree models used lasso-selected features only (12 features)

  Train/Test Split (src/split_data.py):
    - 80/20 stratified split, random_state=42
    - Train: 504,000 rows | Test: 126,000 rows

  Model Evaluation (src/results_evaluation.py):
    - 5-fold stratified cross-validation scored by ROC AUC
    - Final metrics on 20% holdout: AUC, Accuracy, Precision,
      Recall, F1, Confusion Matrix
    - Kaggle test predictions generated via src/generate_submissions.py

  ── REPRODUCIBILITY ────────────────────────────────────────────────
""")

print(f"  Python version : {sys.version.split()[0]}")
print(f"  scikit-learn   : {sklearn.__version__}")
print(f"  pandas         : {pd.__version__}")
print(f"  numpy          : {np.__version__}")

print("""
  Repository     : https://github.com/AidanColvin/machine-learning-midterm-project
  random_state   : 42 (all models and splits)

  To reproduce:
    git clone https://github.com/AidanColvin/machine-learning-midterm-project
    pip install pandas numpy scipy scikit-learn
    python3 src/load-raw-training-data.py
    python3 src/preprocessing.py
    python3 src/split_data.py
    python3 src/feature_engineering.py
    python3 src/results_evaluation.py
    python3 src/generate_submissions.py

  ══════════════════════════════════════════════════════════════════════
""")
