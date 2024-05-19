import numpy as np

def manual_linear_regression(x, y):
    # Adding a column of ones for bias
    x_with_bias = np.c_[x, np.ones(x.shape[0])]

    # Calculating X transpose
    x_transpose = x_with_bias.T

    # Calculating X transpose multiplied by X
    x_transpose_x = np.dot(x_transpose, x_with_bias)

    # Calculating the inverse of X_transpose_X
    x_transpose_x_inv = np.linalg.inv(x_transpose_x)

    # Calculating (X_transpose_X)^-1 multiplied by X transpose
    a = np.dot(x_transpose_x_inv, x_transpose)

    # Calculating the final parameters
    final = np.dot(a, y.reshape(-1, 1))

    # Extracting weights and bias
    weights = final[:-1]
    bias = final[-1]

    return final, weights, bias
