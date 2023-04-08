
import pandas as pd
import numpy as np

DATA_PATH = "data"

def pd_read(path):
    df = pd.read_csv(path)
    X = df.drop("class", axis=1)
    y = df["class"]
    return np.array(X), np.array(y)
