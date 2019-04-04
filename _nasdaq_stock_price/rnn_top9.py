import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout

training_size = 1400
features = 640
num_class = 9
time_steps = 7
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

symbols = ['FB', 'AAPL', 'AMZN', 'NFLX', 'GOOG', 'MSFT', 'IBM', 'ORCL', 'INTC']

# process dataframe in to testing and training data
def process_dataframe(dataframe):

    print(numpy.shape(dataframe.values))
    dataY = dataframe[['FB_Trend_10', 'AAPL_Trend_10', 'AMZN_Trend_10', 'NFLX_Trend_10', 'GOOG_Trend_10', 'MSFT_Trend_10', 'IBM_Trend_10', 'ORCL_Trend_10', 'INTC_Trend_10']].values
    temp = dataframe.drop(columns=['Date', 'FB_Trend_10', 'AAPL_Trend_10', 'AMZN_Trend_10', 'NFLX_Trend_10', 'GOOG_Trend_10', 'MSFT_Trend_10', 'IBM_Trend_10', 'ORCL_Trend_10', 'INTC_Trend_10']).values

    print(numpy.shape(dataY))
    print(numpy.shape(temp))
    dataX = []
    for index in range(len(temp) - time_steps):
        dataX.append(temp[index: index + time_steps])

    dataX = numpy.array(dataX)
    print(numpy.shape(dataX))
    dataY = dataY[time_steps:]

    # normalize the data here in the future if need be

    trainX = dataX[:training_size]
    testX = dataX[training_size:]
    trainY = dataY[:training_size]
    testY = dataY[training_size:]

    return trainX, trainY, testX, testY


def build_model(layer_type="LSTM", layer_num=2, activation_type="sigmoid", loss_type="binary_crossentropy", optimizer_type="rmsprop", dropout_rate=0):
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


datafile = pd.read_csv("./data/combined_top9.csv")
trainX, trainY, testX, testY = process_dataframe(datafile)

# reshaping data for use
trainX = numpy.reshape(trainX, (trainX.shape[0], time_steps, features))
testX = numpy.reshape(testX, (testX.shape[0], time_steps, features))


model = build_model(layer_type=layer_type, layer_num=rnn_layers, activation_type=activation, loss_type=loss, optimizer_type=optimizer, dropout_rate=dropout)
model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(testX, testY))

score = model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

