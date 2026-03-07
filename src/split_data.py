import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_save() -> None:
    """
    Reads the cleaned dataset.
    Splits the data into 80% training and 20% testing sets.
    Saves the splits into the data folder.
    """
    df = pd.read_csv('data/preprocessed/preprocessed-train-data.csv')
    X = df.drop('Heart Disease', axis=1)
    y = df['Heart Disease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train.to_csv('data/preprocessed/X_train.csv', index=False)
    X_test.to_csv('data/preprocessed/X_test.csv', index=False)
    y_train.to_csv('data/preprocessed/y_train.csv', index=False)
    y_test.to_csv('data/preprocessed/y_test.csv', index=False)
    print("Data split completed and saved.")

if __name__ == "__main__":
    split_and_save()