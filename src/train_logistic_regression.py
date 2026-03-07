import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_evaluate() -> None:
    """
    Loads split data and trains Logistic Regression.
    Calculates accuracy and saves the result to CSV.
    """
    X_train = pd.read_csv('data/preprocessed/X_train.csv')
    X_test = pd.read_csv('data/preprocessed/X_test.csv')
    y_train = pd.read_csv('data/preprocessed/y_train.csv')['Heart Disease']
    y_test = pd.read_csv('data/preprocessed/y_test.csv')['Heart Disease']
    
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    result_df = pd.DataFrame({'Model': ['Logistic Regression'], 'Accuracy': [accuracy]})
    result_df.to_csv('data/logistic_regression_results.csv', index=False)
    print("Logistic Regression trained.")

if __name__ == "__main__":
    train_and_evaluate()