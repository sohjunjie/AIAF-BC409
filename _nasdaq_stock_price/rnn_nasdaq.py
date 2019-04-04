import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.utils import plot_model

training_size = 1200
features = 640
num_class = 1
epochs = 100
batch_size = 16

time_steps = 5
layer_type = "GRU" #RNN, GRU or LSTM
rnn_units = 32
rnn_layers = 2
loss = "binary_crossentropy"
activation = "sigmoid"
optimizer = "rmsprop"
dropout = 0


# process dataframe in to testing and training data
def process_dataframe(dataframe):
    dataY = dataframe[["Trend_10"]].values
    temp = dataframe.drop(columns=["Trend_10", "Date"]).values

    dataX = []
    for index in range(len(temp) - time_steps):
        dataX.append(temp[index: index + time_steps])

    dataX = numpy.array(dataX)
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


datafile = pd.read_csv("./data/combined_nasdaq.csv")
trainX, trainY, testX, testY = process_dataframe(datafile)

# reshaping data for use
trainX = numpy.reshape(trainX, (trainX.shape[0], time_steps, features))
testX = numpy.reshape(testX, (testX.shape[0], time_steps, features))

model = build_model(layer_type=layer_type, layer_num=rnn_layers, activation_type=activation, loss_type=loss, optimizer_type=optimizer, dropout_rate=dropout)
model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(testX, testY))

plot_model(model, to_file='model_seq.png', show_shapes=True)

score = model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
