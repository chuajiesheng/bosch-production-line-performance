import numpy as np
import pandas as pd

# load train data
X = np.concatenate([
    pd.read_csv("../input/train_date.csv", index_col=0, dtype=np.float32).values,
    pd.read_csv("../input/train_numeric.csv", index_col=0, dtype=np.float32).values
], axis=1)
y = pd.read_csv("../input/train_numeric.csv", index_col=0, usecols=[0, 969], dtype=np.float32).values.ravel()

