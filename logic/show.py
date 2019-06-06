import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def show_cost(train_costs):
    """
    输出梯度下降过程中损失函数值的变化情况
    """
    sns.set_style("whitegrid")
    plt.plot(train_costs)
    plt.show()


def read_and_show_aqi():
    aqi_data = pd.read_csv("wdbc.csv")
    sns.set(style='whitegrid', context='notebook')
    cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5',
            'C6',
            'C7', 'C8', 'C9', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
    sns.pairplot(aqi_data[cols], size=2.5)
    plt.show()

    cm = np.corrcoef(aqi_data[cols].values.T)
    sns.set(font_scale=1.5)
    sns.heatmap(cm, cbar=True, annot=True, square=True,
                fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
    plt.show()

