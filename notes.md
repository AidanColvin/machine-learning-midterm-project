# Heart Disease Classification — Report Notes & Data Summary

---

## Feature Predictor Strength (Ranked by Correlation with Target)

| Rank | Feature | Correlation | Direction | Strength |
|------|---------|-------------|-----------|----------|
| 1 | Thallium | 60.58% | Positive | ██████████████████ |
| 2 | Chest pain type | 46.07% | Positive | █████████████ |
| 3 | Exercise angina | 44.19% | Positive | █████████████ |
| 4 | Max HR | 44.10% | **Negative** | █████████████ |
| 5 | Number of vessels fluro | 43.86% | Positive | █████████████ |
| 6 | ST depression | 43.06% | Positive | ████████████ |
| 7 | Slope of ST | 41.51% | Positive | ████████████ |
| 8 | Sex | 34.24% | Positive | ██████████ |
| 9 | EKG results | 21.90% | Positive | ██████ |
| 10 | Age | 21.21% | Positive | ██████ |
| 11 | Cholesterol | 8.28% | Positive | ██ |
| 12 | FBS over 120 | 3.36% | Positive | █ |
| 13 | BP | 0.52% | Negative | ~ 0 |

**Key observations:**
- Thallium dominates at 60.6% — single strongest predictor by a wide margin
- BP is essentially useless (0.52%) — confirmed by Lasso feature selection dropping it
- Top 7 features account for most of the model's predictive power
- Max HR is negative — higher heart rate = lower heart disease risk

---

## Model Performance Summary (AUC — ROC)

| Model | CV Folds | Mean AUC | Std | Kaggle Public | Kaggle Private |
|-------|----------|----------|-----|---------------|----------------|
| Logistic Regression (Ridge L2) | 5 | 0.9505 | ±0.0003 | — | — |
| Logistic Regression (Lasso L1) | 10 | 0.9505 | ±0.0007 | — | — |
| SVM (Calibrated LinearSVC) | 5 | 0.9505 | ±0.0003 | — | — |
| Random Forest (100 trees) | 5 | 0.9530 | ±0.0004 | 0.94689 | 0.94496 |
| Gradient Boosting (200 trees) | 5 | 0.9542 | ±0.0004 | — | — |
| Gradient Boosting Deep (300 trees) | 5 | 0.9544 | ±0.0004 | — | — |
| **Best Previous (GB)** | — | — | — | **0.95341** | **0.95168** |

---

## Dataset Description

| Metric | Training | Test |
|--------|----------|------|
| Sample Size | 630,000 | 270,000 |
| Features | 13 | 13 |
| Missing Values | 0 | 0 |
| Duplicate Rows | 0 | 0 |
| Class: Absence | 347,546 (55.2%) | Unknown |
| Class: Presence | 282,454 (44.8%) | Unknown |

---

## Report Writing Notes by Section

---

### 1. Problem Description (20 pts)

**Clarity of problem statement (5 pts)**
- This is a binary classification problem: predict whether a patient has heart disease (Presence/Absence) given 13 clinical features
- The competition uses ROC AUC as the evaluation metric — explain what this means: ability to distinguish between classes at all thresholds
- Frame it: early and accurate detection of heart disease can save lives and reduce healthcare costs

**Dataset description (5 pts)**
- 630,000 training rows, 270,000 test rows, 13 predictor features
- Response variable: `Heart Disease` — binary (Absence = 0, Presence = 1), class split 55.2% / 44.8% (slightly imbalanced but manageable)
- Feature types: 5 continuous (Age, BP, Cholesterol, Max HR, ST depression), 3 binary (Sex, FBS over 120, Exercise angina), 5 ordinal (Chest pain type, EKG results, Slope of ST, Number of vessels fluro, Thallium)
- All continuous features were standardized (mean=0, std=1) during preprocessing
- No missing values in any feature

**Objective (5 pts)**
- Maximize ROC AUC score on Kaggle's held-out test set
- AUC = area under the ROC curve; measures model's ability to rank positive cases above negative ones
- AUC of 1.0 = perfect; 0.5 = random guessing; our best = 0.9517 (private leaderboard)
- Submissions are probability scores (not hard labels), allowing precise AUC calculation

**Relevance to field (5 pts)**
- Heart disease is the leading cause of death globally
- ML-based screening tools can assist clinicians, especially in under-resourced settings
- High AUC means the model reliably flags high-risk patients even before symptoms emerge
- Features like Thallium stress test results and Number of vessels fluoroscopy are standard clinical diagnostics — validating their predictive importance confirms clinical intuition

---

### 2. Methodology (30 pts)

**Overview of approaches (10 pts)**
- Logistic Regression with Ridge (L2) penalty
- Logistic Regression with Lasso (L1) penalty — performs implicit feature selection (dropped BP)
- Logistic Regression with spline features (non-linear transformations of continuous variables)
- Support Vector Machine (LinearSVC with Platt scaling via CalibratedClassifierCV)
- SVM with spline features
- Random Forest (100 trees, max_depth=15)
- Gradient Boosting (200 and 300 trees, various depths)
- Ensemble methods: averaging probability outputs across multiple models

**Rationale for chosen method (10 pts)**
- Gradient Boosting consistently outperformed others in CV AUC (0.9544 vs 0.9505 for logistic)
- GB handles non-linear interactions between features without explicit feature engineering
- Ensemble of GB + RF + Logistic outperforms any individual model by reducing variance
- Logistic regression was computationally cheap and still competitive (0.9505) — useful as ensemble member
- SVM offered no improvement over logistic on this dataset
- Random Forest was memory-constrained on Codespaces (had to reduce to 100 trees)

**Implementation details (5 pts)**
- Preprocessing: StandardScaler fit on training data only, applied to both train and test
- Lasso CV (5-fold) for feature selection → 12/13 features kept (BP dropped, |coef| ≈ 0)
- SplineTransformer (n_knots=5, degree=3) applied to 5 continuous columns → 36 total features
- GridSearchCV over C=[0.001, 0.01, 0.1, 1, 10, 100] for logistic → best C=100
- Stratified K-Fold used throughout to preserve class balance in each fold
- All models trained on 100% of training data for final submission

**Reproducibility (5 pts)**
- Python 3, scikit-learn, pandas, numpy
- `random_state=42` set on all models and CV splits
- Full pipeline in `src/generate_submissions.py`
- Data at `data/preprocessed/preprocessed-train-data.csv` (scaled) and `data/raw/test.csv`
- GitHub repo: `AidanColvin/machine-learning-midterm-project`

---

### 3. Results and Evaluation (25 pts)

**Performance metrics (10 pts)**
- Primary metric: ROC AUC (area under receiver operating characteristic curve)
- Why AUC? The test labels are unknown; AUC rewards well-calibrated probability outputs; robust to class imbalance
- Best Kaggle private score: 0.95168 (Gradient Boosting)
- Best CV AUC: 0.9544 (Gradient Boosting Deep, 300 trees)
- CV scores are very consistent (std ≈ 0.0003–0.0004), indicating stable models

**Analysis of results (10 pts)**
- Gradient Boosting outperforms linear models because it captures non-linear feature interactions (e.g., Thallium × Number of vessels fluro)
- Thallium (0.606 correlation) and Chest pain type (0.461) dominate predictions — consistent with clinical literature
- BP had near-zero correlation (0.005) and was dropped by Lasso — suggests BP alone is not diagnostic without interaction terms
- Max HR being negatively correlated makes clinical sense: lower max HR under stress = reduced cardiac reserve = higher disease risk
- Logistic regression surprisingly competitive (0.9505) suggesting features are largely linearly separable with standardization
- Spline features did not improve performance significantly — likely because tree-based GB already handles non-linearities

**Comparative analysis (5 pts)**
- Use the model comparison table above
- Logistic vs GB: 0.9505 vs 0.9544 (GB better by 0.004)
- Individual GB vs Ensemble: minimal gain from ensembling — GB is already strong
- Old buggy submissions (0.82–0.87): caused by test data scale mismatch — test.csv already preprocessed, rescaling corrupted features

---

### 4. Presentation and Writing Quality (15 pts)

- Use clear section headers: Problem Description → Methodology → Results → Conclusion
- Every table and figure needs a caption and in-text reference ("As shown in Table 2...")
- Define technical terms on first use: AUC, ROC, cross-validation, regularization, L1/L2
- Use the feature correlation table as Figure 1
- Use the model comparison table as Table 1
- Keep sentences short — biostatistics writing is precise and direct

---

### 5. Creativity and Innovation (10 pts)

- Novelty points: spline feature engineering, Lasso-based automatic feature selection, ensemble probability averaging
- Insightfulness: discuss that BP being dropped by Lasso aligns with clinical ambiguity around hypertension as a standalone diagnostic
- Note the scale mismatch bug you discovered and fixed — demonstrates debugging rigor
- Discuss class imbalance (55/45 split) and why it didn't require SMOTE or reweighting at this scale
- Mention computational constraints (Codespaces RAM limits) and how you adapted (reduced RF trees, skipped CV for GB)

---

## Quick Stats to Cite in Report

- Training set: **630,000 rows**, 13 features, 0 missing values
- Test set: **270,000 rows**
- Best Kaggle score: **0.95168** (private), **0.95341** (public)
- Best CV AUC: **0.9544** (Gradient Boosting, 300 trees)
- Logistic CV AUC: **0.9505 ± 0.0003** (5-fold)
- Strongest predictor: **Thallium at 60.6% correlation**
- Weakest predictor: **BP at 0.52% correlation** (dropped by Lasso)
- Models attempted: Logistic Ridge, Logistic Lasso, Logistic+Spline, SVM, SVM+Spline, Random Forest, Gradient Boosting (200), Gradient Boosting (300), Ensembles