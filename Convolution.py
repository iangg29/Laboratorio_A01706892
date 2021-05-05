# Convolytion.py
# Ian García
# TC1001S.120

import numpy as np


def matrix_sum(matrix, kernel):
    """
    Sums & multiplies a matrix with a kernel.
    * We assume that matrix and kernel are the same size.
    * Otherwise an error is expected.
    :param matrix: Matrix num 1.
    :param kernel: Matrix num 2.
    :return output: Sum of multiplication values.
    """
    m_row, m_col = matrix.shape
    output = 0.0
    for i in range(m_row):
        for j in range(m_col):
            output += matrix[i, j] * kernel[i, j]
    return output


def convolution(matrix, kernel):
    """
    Performs convolution of a matrix & a kernel (Both defined by user).
    * We assume matrix does not have padding.
    :param matrix: Main matrix to perform convolution.
    :param kernel: Kernel defined by user in order to make the convolution.
    :return output: Convolution result as a matrix.
    """
    m_row, m_col = matrix.shape
    k_row, k_col = kernel.shape

    o_row = (m_row - k_row) + 1
    o_col = (m_col - k_col) + 1

    output = np.zeros((o_row, o_col))

    for i in range(o_row):
        for j in range(o_col):
            output[i, j] = matrix_sum(matrix[i:i + k_row, j:j + k_col], kernel)
    return output


# EXAMPLE

matrix = np.matrix(
    [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [0, 0, 1, 16, 17, 18], [0, 1, 0, 7, 23, 24], [1, 7, 6, 5, 4, 3]])

kernel = np.matrix([[1, 1, 1], [0, 0, 0], [2, 10, 3]])

print(convolution(matrix, kernel))
