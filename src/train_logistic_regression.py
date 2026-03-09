import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def train_and_evaluate() -> None:
    X_train = pd.read_csv('data/preprocessed/X_train_spline.csv')
    X_test  = pd.read_csv('data/preprocessed/X_test_spline.csv')
    y_train = pd.read_csv('data/preprocessed/y_train.csv')['Heart Disease']
    y_test  = pd.read_csv('data/preprocessed/y_test.csv')['Heart Disease']

    grid = GridSearchCV(
        LogisticRegression(penalty='l2', solver='lbfgs', max_iter=3000, random_state=42),
        param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]},
        cv=5, scoring='roc_auc', n_jobs=1
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"  Best C: {grid.best_params_['C']}  CV AUC: {grid.best_score_:.4f}")

    predictions = best.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    pd.DataFrame({'Model': ['Logistic Regression Ridge'], 'Accuracy': [accuracy]}).to_csv(
        'data/logistic_regression_results.csv', index=False)
    print("Logistic Regression trained.")

if __name__ == "__main__":
    train_and_evaluate()
