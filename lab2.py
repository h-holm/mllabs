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
#                  Kernel functions
# ================================================== #
def linear_kernel(x_vector, y_vector):
    kappa = np.dot(np.transpose(x_vector), y_vector)
    return kappa


def polynomial_kernel(x_vector, y_vector, p=2):
    # kappa = np.power((np.dot(x_vector, y_vector) + 1), p)
    kappa = np.power((np.dot(np.transpose(x_vector), y_vector) + 1), p)
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
            P[i][j] = ti*tj*kappa

    # print(P)
    return P


# ================================================== #
#            Implementation of equation 4
# ================================================== #
# Takes vector alpha as argument.
# Returns a scalar value, effectively implementing equation 4.
def objective(alpha):
    # np.dot()
    scalar = 0
    for i, alpha_i in enumerate(alpha):
        P_i = P[i]
        for j, alpha_j in enumerate(alpha):
            P_i_j = P_i[j]
            scalar += alpha_i * alpha_j * P_i_j

    scalar = 0.5 * scalar - np.sum(alpha)

    return scalar


# ================================================== #
#    Implementation of equality constraint of (10)
# ================================================== #
def zerofun(alpha):
    # if all(elem >= 0 for elem in alpha)
    scalar = np.dot(alpha, targets)

    return scalar


# ================================================== #
#       Calculate b value using equation (7)
# ================================================== #
def calculate_b():
    b = 0.0
    for s_v in support_vectors:
        # if any(([x, y] == [s_v[1], s_v[2]]) for s_v in support_vectors):
        #     b += s_v[0] * s_v[3] * kernel([x, y], [s_v[1], s_v[2]]) - s_v[3]

        # b += s_v[0] * s_v[3] * kernel([x, y], [s_v[1], s_v[2]]) - s_v[3]

        b += s_v[0] * s_v[3] * kernel([s_v[1], s_v[2]], [s_v[1], s_v[2]]) - s_v[3]

    return b


# ================================================== #
#           Indicator function (equation 6)
# ================================================== #
# Implement the indicator function (equation 6) which uses the non-zero αi’s
# together with their xi’s and ti’s to classify new points.
def indicator(x, y):
    ind = 0.0
    for s_v in support_vectors:
        # Missing b?
        ind += s_v[0] * s_v[3] * kernel([x, y], [s_v[1], s_v[2]])
        # ind += s_v[0] * s_v[3] * kernel([x, y], [s_v[1], s_v[2]]) - calculate_b(x, y)

    # ind -= b

    return ind


# ================================================== #
#               Plot input data in 2D
# ================================================== #
def plot_data_and_dec_boundary(class_A, class_B):
    plt.plot([p[0] for p in class_A],
             [p[1] for p in class_A],
             'b.')
    plt.plot([p[0] for p in class_B],
             [p[1] for p in class_B],
             'r.')

    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)

    grid = np.array([[indicator(x, y)
                      for x in xgrid]
                     for y in ygrid])

    plt.contour(xgrid, ygrid, grid,
             (-1.0, 0.0, 1.0),
             colors = ('red', 'black', 'blue'),
             linewidths = (1, 3, 1))

    plt.axis('equal')               # Force same scale on both axes.
    plt.savefig('svm_plot.pdf')     # Save a copy in a file.
    plt.show()                      # Show the plot on the screen.

    return


np.random.seed(100)

# kernel = linear_kernel
kernel = polynomial_kernel

class_A, class_B = generate_n_dist_dataset()
# class_A, class_B = generate_other_dataset()

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

N = inputs.shape[0]
P = np.zeros((N, N))
for i in range(N):
    ti = targets[i]
    xi = inputs[i]
    for j in range(N):
        tj = targets[j]
        xj = inputs[j]
        kappa = kernel(xi, xj)
        P[i][j] = ti*tj*kappa

start = np.zeros(N)
B = [(0, None) for b in range(N)]
XC = {'type': 'eq', 'fun': zerofun}

ret = minimize(objective, start, bounds=B, constraints=XC)
alpha = ret['x']

support_vectors = []
# Extract only non-zero alpha values.
for i in range(len(alpha)):
    # Set threshold for filtering out zero-valued alpha values.
    if abs(alpha[i]) > (10**-5):
        support_vectors.append((alpha[i], inputs[i][0], inputs[i][1], targets[i]))

# print(inputs)
b = calculate_b()
# 1.2693270111146182
# print(b)

plot_data_and_dec_boundary(class_A, class_B)


# if __name__ == '__main__':
#     main()
