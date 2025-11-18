import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):

    df = pd.read_csv(path)

    # Drop Class column from features
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale ALL 30 features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
