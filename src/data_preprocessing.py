import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):

    df = df.drop(
        ["Ad Topic Line","City","Country","Timestamp"],
        axis=1
    )

    X = df.drop("Clicked on Ad", axis=1)
    y = df["Clicked on Ad"]

    return X, y