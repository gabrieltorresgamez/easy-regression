import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR

class LinearRegression:
    def __init__(self, x, y):
        x_r = x.values.reshape(-1, 1)
        y = y.values

        model = LR().fit(x_r, y)
        param = [model.coef_[0], model.intercept_]
        r2 = model.score(x_r, y)

        print("f(x) =", param[0], "* x +", param[1])
        print("R2 =", r2)
        print("")

        plt.title("Linear regression")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(x, y)
        plt.plot(x, param[0] * x + param[1], color = "r")
        plt.show()

        plt.title("Residues")
        plt.scatter(x, y - (param[0] * x + param[1]))
        plt.plot(x, 0 * x, color = "r")
        plt.xlabel("x")
        plt.ylabel("Residues")
        plt.show()

        plt.title("Residues histogram")
        plt.hist(y - (param[0] * x + param[1]), bins = 150, density = True)
        plt.axvline((y - (param[0] * x + param[1])).mean(), color = "r")
        plt.xlabel("Residues")
        plt.ylabel("Frequency")
        plt.show()

        out = "f(x) =" + str(param[0]) + "* x +" + str(param[1]) + "\n" + "R2 = " +  str(r2)
        print(out)