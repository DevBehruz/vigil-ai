import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(path):
    df = pd.read_csv(path)

    train_df = df[df['split'] == 'train'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        train_df['text'],
        train_df['label'],
        test_size=1086,
        random_state=42,
        stratify=train_df['label']
    )

    return X_train, X_test, y_train, y_test
