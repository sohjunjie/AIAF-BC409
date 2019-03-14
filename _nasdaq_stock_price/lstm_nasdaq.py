import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

training_size = 1000
features = 60
num_class = 1
time_steps = 1
lstm_units = 32

epochs = 20
batch_size = 1


# process dataframe in to testing and training data
def process_dataframe(dataframe):
    dataY = dataframe["Momentum"].values
    dataX = dataframe.drop(columns=["Momentum"]).values

    trainX = dataX[:training_size]
    testX = dataX[training_size:]
    trainY = dataY[:training_size]
    testY = dataY[training_size:]

    return trainX, trainY, testX, testY


datafile = pd.read_csv("./data/combined_nasdaq.csv")
trainX, trainY, testX, testY = process_dataframe(datafile)

# normalize the data here in the future if need be

# reshaping data for use
trainX = numpy.reshape(trainX, (trainX.shape[0], time_steps, features))
testX = numpy.reshape(testX, (testX.shape[0], time_steps, features))

# build model
model = Sequential()
model.add(LSTM(lstm_units, return_sequences=True, input_shape=(batch_size, features)))
model.add(LSTM(lstm_units, return_sequences=True))
model.add(LSTM(lstm_units, return_sequences=True))
model.add(LSTM(lstm_units, return_sequences=True))
model.add(LSTM(lstm_units, return_sequences=True))
model.add(LSTM(lstm_units, return_sequences=True))
model.add(LSTM(lstm_units, return_sequences=True))
model.add(LSTM(lstm_units, return_sequences=True))
model.add(LSTM(lstm_units))
model.add(Dense(num_class, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(testX, testY))

score = model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

