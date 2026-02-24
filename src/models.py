import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class MultiOutputModel:
    def __init__(self, base_estimator=None, random_state: int = 42):
        self.base_estimator = base_estimator or RandomForestRegressor(
            n_estimators=200, random_state=random_state
        )
        self.model = MultiOutputRegressor(self.base_estimator)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, Y):
        pred = self.predict(X)
        rmse = float(np.sqrt(mean_squared_error(Y, pred)))
        return {"rmse": rmse}
