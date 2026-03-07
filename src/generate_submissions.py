import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

def create_submissions() -> None:
    """
    given nothing
    trains models on full dataset, prints comparison, saves submissions
    skips heavy models that exceed codespace cpu limits
    """
    train_df = pd.read_csv('data/preprocessed/preprocessed-train-data.csv')
    X_train  = train_df.drop(['Heart Disease', 'id', 'is_outlier'], axis=1, errors='ignore')
    y_train  = train_df['Heart Disease']

    test_df  = pd.read_csv('data/raw/test.csv')
    test_ids = test_df['id']
    X_test   = test_df.drop(['id', 'is_outlier'], axis=1, errors='ignore')
    X_test   = X_test.reindex(columns=X_train.columns, fill_value=0)

    os.makedirs('data/submissions', exist_ok=True)

    models = {
        'logistic_regression': LogisticRegression(max_iter=2000),
        'svm'                : CalibratedClassifierCV(LinearSVC(random_state=42, dual=False), cv=3),
    }

    results = []

    for name, model in models.items():
        model_name = name.replace('_', ' ').title()
        print(f"\n--- Processing {model_name} ---")

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"Cross-Validated ROC AUC: {cv_scores.mean():.4f}")

        print("Training on full dataset...")
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]

        output_file = f'data/submissions/submission_{name}.csv'
        pd.DataFrame({'id': test_ids, 'Heart Disease': probabilities}).to_csv(output_file, index=False)
        print(f"Saved {output_file}")

        results.append({'Model': model_name, 'ROC AUC': round(cv_scores.mean(), 4)})

    results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)
    results_df.to_csv('data/preprocessed/submission_comparison.csv', index=False)

    print("\n  ── Submission Model Comparison ─────────────────")
    print(f"     {'Model':<30} {'ROC AUC':>10}")
    print(f"     {'─────':<30} {'───────':>10}")
    for _, row in results_df.iterrows():
        bar = "█" * int(row['ROC AUC'] * 30)
        print(f"     {row['Model']:<30} {row['ROC AUC']:>10.4f}  {bar}")

    best = results_df.iloc[0]
    print(f"\n  ✔  Best submission: submission_{best['Model'].lower().replace(' ', '_')}.csv ({best['ROC AUC']:.4f} ROC AUC)")
    print(f"  ── Comparison saved to: data/preprocessed/submission_comparison.csv\n")

if __name__ == "__main__":
    create_submissions()
