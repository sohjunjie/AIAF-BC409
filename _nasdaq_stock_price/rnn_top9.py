import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from sklearn .linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import matplotlib.pyplot as plt

training_ratio = 0.8 # the rest will be split half half between validate and test data
num_class = 9
time_steps = 10
epochs = 50
batch_size = 16

time_steps = 10
layer_type = "LSTM" # RNN, GRU or LSTM
rnn_units = 32
rnn_layers = 3
loss = "binary_crossentropy"
activation = "sigmoid"
optimizer = "rmsprop"
dropout = 0.1

features = 600 # original 640

symbols = ['FB', 'AAPL', 'AMZN', 'NFLX', 'GOOG', 'MSFT', 'IBM', 'ORCL', 'INTC']

# process dataframe in to testing and training data
def process_dataframe(num_features):
    df = pd.read_csv("./data/combined_top9.csv")
    dataY = df[['FB_Trend_10', 'AAPL_Trend_10', 'AMZN_Trend_10', 'NFLX_Trend_10', 'GOOG_Trend_10', 'MSFT_Trend_10', 'IBM_Trend_10', 'ORCL_Trend_10', 'INTC_Trend_10']].values
    tempX = df.drop(columns=['Date', 'FB_Trend_10', 'AAPL_Trend_10', 'AMZN_Trend_10', 'NFLX_Trend_10', 'GOOG_Trend_10', 'MSFT_Trend_10', 'IBM_Trend_10', 'ORCL_Trend_10', 'INTC_Trend_10']).values


    # nomalizing data
    min_max_scaler = preprocessing.MinMaxScaler()
    tempX = min_max_scaler.fit_transform(tempX)

    df2 = pd.read_csv("./data/combined_nasdaq.csv")
    tempY = df2[["Trend_10"]].values

    # feature selection
    #  Create a logistic regression estimator
    logreg = LogisticRegression()

    # Use RFECV to pick best features, using Stratified Kfold
    rfe = RFE(logreg, num_features)
    tempX = rfe.fit_transform(tempX, tempY)

    dataX = []
    for index in range(len(tempX) - time_steps):
        dataX.append(tempX[index: index + time_steps])
    dataY = dataY[time_steps:]

    validating_ratio = training_ratio + (1-training_ratio) * 0.5
    trainX, validateX, testX = np.split(dataX, [int(training_ratio*len(dataX)), int(validating_ratio*len(dataX))])
    trainY, validateY, testY = np.split(dataY, [int(training_ratio*len(dataY)), int(validating_ratio*len(dataY))])

    return trainX, trainY, validateX, validateY, testX, testY


def build_model(features, layer_type="LSTM", layer_num=2, activation_type="sigmoid", loss_type="binary_crossentropy", optimizer_type="rmsprop", dropout_rate=0):
    model = Sequential()
    if layer_type == "GRU":
        rnn_layer = GRU
    elif layer_type == "LSTM":
        rnn_layer = LSTM
    else:
        rnn_layer = SimpleRNN

    model.add(rnn_layer(rnn_units, return_sequences=True, input_shape=(time_steps, features)))
    if layer_num < 2:
        layer_num = 2
    for i in range(layer_num - 2):
        model.add(rnn_layer(rnn_units, return_sequences=True))

    model.add(rnn_layer(rnn_units))
    if dropout_rate != 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_class, activation=activation_type))

    model.compile(loss=loss_type, optimizer=optimizer_type, metrics=['accuracy'])
    return model


trainX, trainY, validateX, validateY, testX, testY = process_dataframe(features)

# reshaping data for use
trainX = np.reshape(trainX, (trainX.shape[0], time_steps, features))
validateX = np.reshape(validateX, (validateX.shape[0], time_steps, features))
testX = np.reshape(testX, (testX.shape[0], time_steps, features))


model = build_model(features , layer_type=layer_type, layer_num=rnn_layers, activation_type=activation, loss_type=loss, optimizer_type=optimizer, dropout_rate=dropout)
model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(validateX, validateY))

score = model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


test_pred = model.predict(testX)

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

for i in range(len(testY[0])):
    for j in range(len(testY)):
        if testY[j][i] == round(test_pred[j][i]):
            if testY[j][i] == 1:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if testY[j][i] == 1:
                false_neg += 1
            else:
                false_pos += 1

print("True positive: " + str(true_pos))
print("True negative: " + str(true_neg))
print("False positive: " + str(false_pos))
print("False negative: " + str(false_neg))

