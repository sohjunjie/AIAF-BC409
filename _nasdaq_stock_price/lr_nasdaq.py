import matplotlib.pyplot as plt
import numpy as np
from sklearn .linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
import pandas as pd

training_size = 1200
datafile = pd.read_csv("./data/NDAQ_cleaned.csv")

dataY = datafile[["Trend_10"]].values
dataX = datafile.drop(columns=["Trend_10", "Date"]).values

# remove variable
#  Create a logistic regression estimator
logreg = LogisticRegression()

# Use RFECV to pick best features, using Stratified Kfold
rfe = RFE(logreg, 10)
dataX = rfe.fit_transform(dataX, dataY)

print(dataX.shape[1])

trainX = dataX[:training_size]
testX = dataX[training_size:]
trainY = dataY[:training_size]
testY = dataY[training_size:]
print(np.shape(trainX))
print(np.shape(testX))
print(np.shape(trainY))
print(np.shape(testY))

regr = LogisticRegression()
regr.fit(trainX, trainY)

test_pred = regr.predict(testX)

# The coefficients
print(classification_report(testY, test_pred))

# Plot outputs
plt.plot(testY,  color='black')
plt.plot(test_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
