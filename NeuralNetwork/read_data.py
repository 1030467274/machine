import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


def read():
    data = pd.read_csv("train.csv")
    data = data.drop(["PassengerId", "Cabin", "Name", "Ticket"], axis=1)
    labe1 = data["Survived"].values.reshape(-1, 1)
    age = int(np.mean(data["Age"].fillna(0).values))
    value = {"Age": age, "Embarked": "un"}
    data = data.fillna(value)
    cols = ["Age", "SibSp", "Parch", "Fare"]
    # 将数据中的字符串，转换为数值
    coder = LabelEncoder()
    ont = OneHotEncoder()
    temp = data[["Sex", "Embarked"]].apply(lambda item:coder.fit_transform(item))

    temp = ont.fit_transform(temp.values).astype(int).toarray()
    data = data[cols].apply(lambda item: (item - np.min(item)) / (np.max(item)-np.min(item)), axis=0).values
    data = np.hstack((data, temp))

    return data, labe1
