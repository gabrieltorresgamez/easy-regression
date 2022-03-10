import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, x, y):
        data = pd.DataFrame(
            {'x': x,
            'y': y
            }
        )
        train, test = train_test_split(data, test_size=0.2)
        x_r = train["x"].values.reshape(-1, 1)
        y = train["y"].values
        x_t = test["x"].values.reshape(-1, 1)
        y_t = test["y"].values

        model = LR().fit(x_r, y)
        r2 = model.score(x_t, y_t)

        x_predict = np.linspace(np.min(x_r), np.max(x_r), 50).reshape(-1, 1)
        y_predict = model.predict_proba(x_predict)[:,1]

        plt.title("Logistic regression")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(data["x"], data["y"])
        plt.plot(x_predict, y_predict, color = "r")
        plt.show()

        out = "R2 = " + str(r2)
        print(out)