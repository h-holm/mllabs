import random, math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sklearn.datasets as dt


np.random.seed(100)

# ================================================== #
#                Generate data sets
# ================================================== #
def generate_n_dist_dataset_original():
    class_A = np.concatenate(
                             (np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
                              np.random.randn(10, 2) * 0.2 + [-1.5, 0.5])
                            )
    class_B = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

    return class_A, class_B


def generate_n_dist_dataset_one_cluster():
    class_A = np.concatenate(
                             (np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
                              np.random.randn(10, 2) * 0.2 + [1.5, 0.5])
                            )
    class_B = np.random.randn(20, 2) * 0.2 + [1.5, 0.5]

    return class_A, class_B


def generate_n_dist_dataset_inseparable():
    class_A = np.concatenate(
                             (np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
                              np.random.randn(10, 2) * 0.2 + [-1.5, 0.5])
                            )
    class_B = np.random.randn(20, 2) * 0.2 + [0.0, 0.5]

    return class_A, class_B


def create_inputs_and_targets_arrays(A_data, B_data):
    datapoints = np.concatenate((A_data, B_data))
    target_values = np.concatenate(
                             (np.ones(A_data.shape[0]),
                             -np.ones(B_data.shape[0])))

    # Number of rows (samples).
    N = datapoints.shape[0]

    # Randomly reorder the samples.
    permute = list(range(N))
    random.shuffle(permute)
    datapoints = datapoints[permute, :]
    target_values = target_values[permute]

    return datapoints, target_values

# ================================================== #
#                  Kernel functions
# ================================================== #
def linear_kernel(x_vector, y_vector):
    kappa = np.dot(np.transpose(x_vector), y_vector)
    return kappa


def polynomial_kernel(x_vector, y_vector, p=3):
    kappa = np.power((np.dot(np.transpose(x_vector), y_vector) + 1), p)
    return kappa


def RBF_kernel(x_vector, y_vector, sigma=0.5):
    diff = np.subtract(x_vector, y_vector)
    kappa = math.exp((-np.dot(diff, diff)) / (2 * sigma * sigma))
    return kappa


# ================================================== #
#              Instantiate vector Pij
# ================================================== #
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

    return P


# ================================================== #
#            Implementation of equation 4
# ================================================== #
def objective(alpha):
    tmp_vector = np.dot(alpha, P)
    tmp_scalar = np.dot(alpha, tmp_vector)
    scalar = 1/2 * tmp_scalar - np.sum(alpha)

    return scalar


# ================================================== #
#    Implementation of equality constraint of (10)
# ================================================== #
def zerofun(alpha):
    return np.dot(alpha, targets)


# ================================================== #
#       Calculate b value using equation (7)
# ================================================== #
def calculate_b():
    b = 0.0

    first_sv = support_vectors[0]
    first_sv_t = first_sv[3]

    for sv in support_vectors:
        b += sv[0] * sv[3] * kernel(np.asarray([first_sv[1], first_sv[2]]), np.asarray([sv[1], sv[2]]))

    b -= first_sv[3]

    return b


# ================================================== #
#           Indicator function (equation 6)
# ================================================== #
def indicator(x, y):
    ind = 0.0
    for sv in support_vectors:
        ind += sv[0] * sv[3] * kernel(np.asarray([x, y]), np.asarray([sv[1], sv[2]]))

    ind -= b

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
    ygrid = np.linspace(-5, 5)
    # xgrid = np.linspace(-5, 5)
    # ygrid = np.linspace(-4, 4)

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


# ================================================== #
#      Extract support vectors using threshold
# ================================================== #
def extract_support_vectors(alpha, threshold=10**-5):
    support_vectors = []
    for i in range(len(alpha)):
        # Set threshold for filtering out zero-valued alpha values.
        if abs(alpha[i]) > (10**-5):
        # if abs(alpha[i]) > (10**-5) and abs(alpha[i]) <= C:
            support_vectors.append((alpha[i], inputs[i][0], inputs[i][1], targets[i]))

    return support_vectors


def main():
    global kernel
    # kernel = linear_kernel
    # kernel = polynomial_kernel
    kernel = RBF_kernel

    # Higher C-value equals lower slack.
    C = 100

    # Our "in-house" created dataset.
    # class_A, class_B = generate_n_dist_dataset_original()
    class_A, class_B = generate_n_dist_dataset_inseparable()
    # class_A, class_B = generate_n_dist_dataset_one_cluster()

    # Create data structures for the input datasets and their corresponding
    # target values (ti E {-1, 2}). Both are np.arrays.
    global inputs
    global targets
    inputs, targets = create_inputs_and_targets_arrays(class_A, class_B)

    # Instantiate P to save computational power and set it to being global.
    global P
    P = instantiate_P(inputs, targets, kernel)

    N = inputs.shape[0]

    # Set a zeroed vector as our initial guesses.
    start = np.zeros(N)
    # Bounds.
    B = [(0, C) for b in range(N)]
    # Extra constraints. In this case, of type "equality" using our zerofun().
    XC = {'type': 'eq', 'fun': zerofun}

    ret = minimize(objective, start, bounds=B, constraints=XC)
    print('Success: ', ret['success'])
    print('Message: ', ret['message'])
    alpha = ret['x']

    print(alpha.astype(int))

    # Extract only non-zero alpha values.
    global support_vectors
    support_vectors = extract_support_vectors(alpha)

    global b
    b = calculate_b()

    plot_data_and_dec_boundary(class_A, class_B)


if __name__ == '__main__':
    main()
