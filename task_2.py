import time
import numpy as np
import pandas as pd


def execution_time_measure(unit="second"):
    """
    Параметризованный декоратор, который принимает на вход единицу измерения
    (секунда или миллисекунда), и печатает время выполнения декорированной функции.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            result = func(*args, **kwargs)

            end_time = time.perf_counter()

            elapsed_time = end_time - start_time

            if unit == "millisecond":
                elapsed_time *= 1000
                print(
                    f"Время выполнения функции '{func.__name__}': {elapsed_time} миллисекунд."
                )
            else:
                print(
                    f"Время выполнения функции '{func.__name__}': {elapsed_time} секунд."
                )

            return result

        return wrapper

    return decorator


@execution_time_measure(unit="millisecond")
def diagonals_sum(matrix):
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
    elif isinstance(matrix, list):
        matrix = np.array(matrix)

    primary_diagonal_sum = np.trace(matrix)
    secondary_diagonal_sum = np.trace(np.fliplr(matrix))

    return primary_diagonal_sum + secondary_diagonal_sum


diagonals_sum(
    np.array([[1, 2, 3, 4], [5, 6.0, 7.0, 8], [9, 10.0, 11.0, 12], [13, 14, 15, 16]])
)
