import matplotlib.pyplot as plt
import numpy as np


def calculate_cross_entropy(x):
    return -np.log(x)


def calculate_focal(x):
    alpha = 0.25
    gamma = 2

    x = calculate_cross_entropy(x)
    return alpha * (x * ((1 - x) ** gamma))


def plot_losses():
    x = np.arange(0.01, 1.01, 0.01)
    y_ce = [calculate_cross_entropy(i) for i in x]
    focal = [calculate_focal(i) for i in x]

    plt.plot(x, y_ce)
    plt.plot(x, focal)
    plt.show()
    print()

if __name__ == '__main__':
    plot_losses()
