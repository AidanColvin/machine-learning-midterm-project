import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

def create_submissions():
    """
    Trains four machine learning models using cross-validation.
    Generates Kaggle submission CSV files with probability predictions.
    """
    if not os.path.exists('data/raw/test.csv'):
        print("Error: data/test.csv not found.")
        return

    print("Loading training data...")
    train_df = pd.read_csv('data/preprocessed/preprocessed-train-data.csv')
    X_train = train_df.drop(['Heart Disease', 'id', 'is_outlier'], axis=1, errors='ignore')
    y_train = train_df['Heart Disease']

    print("Loading test data...")
    test_df = pd.read_csv('data/raw/test.csv')
    X_test = test_df.drop(['id', 'is_outlier'], axis=1, errors='ignore')
    test_ids = test_df['id']

    # Use a fast LinearSVC wrapped in a probability calibrator
    fast_svm = CalibratedClassifierCV(LinearSVC(random_state=42, dual=False), cv=5)

    models = {
        'logistic_regression': LogisticRegression(max_iter=2000),
        'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'svm': fast_svm,
        'gradient_boosting': GradientBoostingClassifier(random_state=42)
    }

    for name, model in models.items():
        model_name = name.replace('_', ' ').title()
        print(f"--- Processing {model_name} ---")
        
        # Calculate cross-validated ROC AUC score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        print(f"Cross-Validated ROC AUC Score: {cv_scores.mean():.4f}")

        # Train model and generate decimal probabilities
        print("Training on full dataset...")
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]

        # Save formatted Kaggle submission file
        submission_df = pd.DataFrame({'id': test_ids, 'Heart Disease': probabilities})
        output_file = f'data/submissions/submission_{name}.csv'
        submission_df.to_csv(output_file, index=False)
        print(f"Saved {output_file}\n")

if __name__ == "__main__":
    create_submissions()