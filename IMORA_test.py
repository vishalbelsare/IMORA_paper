import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
from os.path import dirname, join

from IMORA.imora import IMORA

if __name__ == '__main__':
    racine_path = dirname(__file__)
    # Create a random dataset
    data = pd.read_csv(join(racine_path + '/Data/Bias_correction_ucl.csv'), index_col=1,
                       parse_dates=True)
    data.dropna(inplace=True)
    target = ['Next_Tmax', 'Next_Tmin']
    y = data[target]
    X = data.drop(columns=target)
    features = X.columns

    y_train = y.loc[y.index < '01-01-2015'].values
    y_test = y.loc[y.index >= '01-01-2015'].values
    X_train = X.loc[X.index < '01-01-2015'].values
    X_test = X.loc[X.index >= '01-01-2015'].values

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
    regr_imora.fit(X_train, y_train, features)

    # Predict on new data
    y_multirf = regr_multirf.predict(X_test)
    y_rf = regr_rf.predict(X_test)
    y_imora = regr_imora.predict(X_test)

    print('MSE')
    print('---')
    for i in range(y_test.ndim):
        print('Dimension ', str(i))
        print('MultiOutputRegressor mse:', np.mean((y_test[:, i] - y_multirf[:, i]) ** 2))
        print('Random Forest mse:', np.mean((y_test[:, i] - y_rf[:, i]) ** 2))
        y_imora[:, i][y_imora[:, i] == 0] = y_train.mean(axis=0)[i]
        print('IMORA mse:', np.mean((y_test[:, i] - y_imora[:, i]) ** 2))
