import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def read():
    data = pd.read_csv("data/train.csv")
    data = data.drop(["Cabin"], axis=1)
    # 处理数据中的空值
    age = int(np.mean(data["Age"].fillna(0).values))
    value = {"Age": age, "Embarked": "un"}
    data = data.fillna(value)
    cols = ["Age", "SibSp", "Parch", "Fare"]
    # 将数据中的字符串，转换为数值
    coder = LabelEncoder()
    temp = data[["Sex", "Embarked"]].apply(lambda item: coder.fit_transform(item)).values
    data = np.hstack((data[cols].values, temp))
    data = pd.DataFrame(data, columns=["Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"])
    # 展示数据之间的关系
    cm = np.corrcoef(data.values.T)
    sns.set(font_scale=1.5)
    sns.heatmap(cm, cbar=True, annot=True, square=True,
                fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
    plt.show()
    print(data)


read()