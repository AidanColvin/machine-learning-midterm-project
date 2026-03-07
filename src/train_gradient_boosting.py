import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate() -> None:
    """
    Loads split data and trains Gradient Boosting.
    Calculates accuracy and saves the result to CSV.
    """
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')['Heart Disease']
    y_test = pd.read_csv('data/y_test.csv')['Heart Disease']
    
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    result_df = pd.DataFrame({'Model': ['Gradient Boosting'], 'Accuracy': [accuracy]})
    result_df.to_csv('data/gradient_boosting_results.csv', index=False)
    print("Gradient Boosting trained.")

if __name__ == "__main__":
    train_and_evaluate()