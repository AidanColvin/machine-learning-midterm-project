import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def train_and_evaluate() -> None:
    """
    Loads split data and trains a Linear Support Vector Machine.
    Calculates accuracy and saves the result to CSV.
    """
    X_train = pd.read_csv('data/preprocessed/X_train.csv')
    X_test = pd.read_csv('data/preprocessed/X_test.csv')
    y_train = pd.read_csv('data/preprocessed/y_train.csv')['Heart Disease']
    y_test = pd.read_csv('data/preprocessed/y_test.csv')['Heart Disease']
    
    model = LinearSVC(random_state=42, dual=False)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    result_df = pd.DataFrame({'Model': ['Linear Support Vector Machine'], 'Accuracy': [accuracy]})
    result_df.to_csv('data/preprocessed/svm_results.csv', index=False)
    print("Linear SVM trained.")

if __name__ == "__main__":
    train_and_evaluate()