import copy

import numpy as np

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def get_forecasting_val(values, llen, eps, lback):
    '''
        This function use a LSTM Neuran Network in order to forecast
    values for based on historical data
    Params:
    ------
    values: list - historical values
    llen: int    - length of forecasted values list
    eps: int     - number of training epochs
    lback: int   - lookback for LSTM

    Returns:
    -------
    prediction: list - forecasted values
    '''

    look_back = lback
    nepochs = eps
    batch_size = 8

    dataset = copy.deepcopy(values)
    for i in range(len(dataset)):
        dataset[i] = [dataset[i]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train = np.array(dataset)
    trainX, trainY = create_dataset(train, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=nepochs, batch_size=batch_size, verbose=0)

    l = trainX[len(trainX) - 1]
    s = [list(x) for x in l]
    prediction = []
    for it in range(llen + 1):
        t = np.array([s])
        pred = model.predict(t)
        predt = scaler.inverse_transform(pred)
        k = []
        for i in range(1, len(s)):
            k.append(s[i])
        k.append(list(pred[0]))
        s = k
        prediction.append(int(predt))
    del prediction[0]

    return prediction
