import unittest
import numpy as np
import pandas as pd

from task_1 import diagonals_sum


class TestDiagonalsSum(unittest.TestCase):

    def test_list_input(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected_result = 30
        self.assertEqual(diagonals_sum(matrix), expected_result)

    def test_numpy_array_input(self):
        matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )
        expected_result = 68
        self.assertEqual(diagonals_sum(matrix), expected_result)

    def test_pandas_dataframe_input(self):
        matrix = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]})
        expected_result = 30
        self.assertEqual(diagonals_sum(matrix), expected_result)


if __name__ == "__main__":
    unittest.main()
