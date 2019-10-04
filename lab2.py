import random, math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ================================================== #
#      Generate Points with Normal Distribution
# ================================================== #
def generate_n_dist_dataset():
    class_A = np.concatenate(
                             (np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
                              np.random.randn(10, 2) * 0.2 + [-1.5, 0.5])
                            )
    class_B = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

    return class_A, class_B


# ================================================== #
#               Linear Kernel function
# ================================================== #
def linear_kernel(x_vector, y_vector):
    # kappa = np.dot(np.transpose(x_vector), y_vector)
    kappa = np.dot(x_vector, y_vector) + 1
    # kappa = np.dot(x_vector, y_vector)
    return kappa


# ================================================== #
#              Instantiate vector Pij
# ================================================== #
# TODO: NOT SURE ABOUT THIS.
def instantiate_P(inputs, targets, kernel):
    N = inputs.shape[0]
    P = np.zeros((N, N))
    for i in range(N):
        ti = targets[i]
        xi = inputs[i]
        for j in range(N):
            tj = targets[j]
            xj = inputs[j]
            kappa = kernel(xi, xj)
            # print(xi)
            # print(xj)
            # print(ti)
            # print(tj)
            # print(kappa)
            P[i][j] = ti*tj*kappa

    # print(P)
    return P


# ================================================== #
#            Implementation of equation 4
# ================================================== #
def objective(alpha, INPUT_TO_OBJ):
    np.dot(P)- np.sum(alpha)
    np.dot()
    np.sum()
    return


# 0 <= alphai <= C for all i     AND    SUMi(alphai * ti) = 0
def zerofun():
    np.dot()
    return


# ================================================== #
#                  Indicator function
# ================================================== #
# Implement the indicator function (equation 6) which uses the non-zero αi’s
# together with their xi’s and ti’s to classify new points.
def indicator(svs, xs, ys, kernel):
    ind = 0.0

    for i in range(len(svs)):
        ind += (svs[i])[0] * (svs[i])[3] * kernel([x, y], [(svs[i])[1], (svs[i])[2]])

    return ind


# ================================================== #
#               Plot input data in 2D
# ================================================== #
def plot_input_data(class_A, class_B):
    plt.plot([p[0] for p in class_A],
             [p[1] for p in class_A],
             'b.')
    plt.plot([p[0] for p in class_B],
             [p[1] for p in class_B],
             'r.')

    plt.axis('equal')           # Force same scale on both axes.
    plt.savefig('svmplot.pdf')  # Save a copy in a file.
    plt.show()                  # Show the plot on the screen.

    return


# You can use global variables for other things that the function needs
# (t and K values).
N = inputs.shape[0]
P = np.zeros((N, N))
for i in range(N):
    ti = targets[i]
    xi = inputs[i]
    for j in range(N):
        tj = targets[j]
        xj = inputs[j]
        kappa = kernel(xi, xj)
        # print(xi)
        # print(xj)
        # print(ti)
        # print(tj)
        # print(kappa)
        P[i][j] = ti*tj*kappa


def main():
    np.random.seed(100)

    kernel = linear_kernel
    # kernel = poly_kernel

    class_A, class_B = generate_n_dist_dataset()
    # class_A, class_B = generate_noisy_dataset()

    inputs = np.concatenate((class_A, class_B))
    targets = np.concatenate(
                             (np.ones(class_A.shape[0]),
                             -np.ones(class_B.shape[0])))

    # Number of rows (samples).
    N = inputs.shape[0]

    # Randomly reorder the samples.
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    # P = instantiate_P(inputs=inputs, targets=targets, kernel=linear_kernel)

    print(P)

    start = np.zeros((N, N))
    B = [(0, None) for b in range(N)]
    XC = {'type': 'eq', 'fun': zerofun}

    ret = minimize(objective, start, bounds=B, constraints=XC, INPUT_TO_OBJ=P)

    plot_input_data(class_A, class_B)

    # plot_decision_boundary()
    # xgrid = np.linspace(-5, 5)
    # ygrid = np.linspace(-4, 4)
    #
    # grid = np.array([[indicator(x, y)
    #                   for x in xgrid]
    #                  for y in ygrid])
    #
    # plt.contour(xgrid, ygrid, grid,
    #             (-1.0, 0.0, 1.0),
    #             colors = ('red', 'black', 'blue'),
    #             linewidths = (1, 3, 1))


if __name__ == '__main__':
    main()
