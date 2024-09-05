import numpy as np
import pandas as pd



def ridge_based_brute_linear_regression(x, y, lambda_reg=1e-5):
    # Adding a column of ones for bias
    x_with_bias = np.c_[x, np.ones(x.shape[0])]

    # Calculating X transpose
    x_transpose = x_with_bias.T

    # Calculating X transpose multiplied by X
    x_transpose_x = np.dot(x_transpose, x_with_bias)

    # Adding regularization term (lambda_reg * I)
    regularization_term = lambda_reg * np.eye(x_transpose_x.shape[0])
    x_transpose_x_reg = x_transpose_x + regularization_term

    # Calculating the inverse of X_transpose_X with regularization
    x_transpose_x_inv = np.linalg.inv(x_transpose_x_reg)

    # Calculating (X_transpose_X)^-1 multiplied by X transpose
    a = np.dot(x_transpose_x_inv, x_transpose)

    # Converting y to a NumPy array and reshaping it
    y_array = y.to_numpy().reshape(-1, 1)

    # Calculating the final parameters
    final = np.dot(a, y_array)

    # Extracting weights and bias
    weights = final[:-1]
    bias = final[-1]

    return final, weights, bias

# Implementation based function
def collinearity_based_brute_linear_regression(x, y):
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

    # Converting y to a NumPy array and reshaping it
    y_array = y.to_numpy().reshape(-1, 1)

    # Calculating the final parameters
    final = np.dot(a, y_array)

    # Extracting weights and bias
    weights = final[:-1]
    bias = final[-1]

    return final, weights, bias
