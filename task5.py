from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np


def train_and_test_model(n_samples=1000, n_features=20, random_state=0, test_size=0.2):
    rng = np.random

    X = rng.randn(n_samples, n_features)
    y = rng.poisson(lam=np.exp(X[:, 5]) / 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    regressor = DecisionTreeRegressor(criterion="poisson", random_state=random_state)
    regressor.fit(X_train, y_train)

    return regressor, X_test, y_test
