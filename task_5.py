import unittest
from task5 import train_and_test_model


class TestModelTraining(unittest.TestCase):
    def test_regressor_training(self):
        # Уменьшила выборку для теста
        regressor, X_test, y_test = train_and_test_model(
            n_samples=100, n_features=20, random_state=0
        )

        self.assertIsNotNone(regressor, "Regressor should not be None.")
        self.assertTrue(
            hasattr(regressor, "predict"), "Regressor should have a predict method."
        )

        predictions = regressor.predict(X_test)
        self.assertEqual(
            len(predictions),
            len(y_test),
            "Predictions and y_test should have the same length.",
        )


if __name__ == "__main__":
    unittest.main()
