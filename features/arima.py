import typing as t
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def arima_predict(train_data):
    """
    Returns next predicted value
    """
    if type(train_data) == pd.Series:
        data = list(train_data)
    elif type(train_data) == np.array:
        data = list(train_data)
    else:
        data = train_data
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    return output[0]
