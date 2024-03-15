import numpy as np
import pandas as pd


def diagonals_sum(matrix):

    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
    elif isinstance(matrix, list):
        matrix = np.array(matrix)

    primary_diagonal_sum = np.trace(matrix)

    secondary_diagonal_sum = np.trace(np.fliplr(matrix))

    return primary_diagonal_sum + secondary_diagonal_sum
