import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from IMORA.imora import IMORA

if __name__ == '__main__':
    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y += (0.5 - rng.rand(*y.shape))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=400, test_size=200, random_state=4)

    max_depth = 30

    # Multi Random Forest
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                              max_depth=max_depth,
                                                              random_state=0))
    regr_multirf.fit(X_train, y_train)

    # Random Forest
    regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                    random_state=2)
    regr_rf.fit(X_train, y_train)

    # IMORA
    regr_imora = IMORA()
    regr_imora.fit(X_train, y_train)

    # Predict on new data
    y_multirf = regr_multirf.predict(X_test)
    y_rf = regr_rf.predict(X_test)