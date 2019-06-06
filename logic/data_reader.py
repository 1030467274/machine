import pandas as pd
import numpy as np


def read_data():
    data = pd.read_csv("wdbc.csv")
    data = data.sample(frac=1.0)
    cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5',
            'C6',
            'C7', 'C8', 'C9', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
    x = data[cols]
    y = data['diagnosis']

    x = x.apply(lambda item: (item-np.min(item))/(np.max(item)-np.min(item)), axis=1)
    y = y.apply(lambda label: 1 if label == 'M' else 0)
    return x.values, y.values.reshape(-1, 1)

