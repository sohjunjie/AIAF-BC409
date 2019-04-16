import numpy as np
import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
from sklearn import svm
import numpy as np
from csv import reader 


file_name = 'data.csv'
#sheet = 'Sheet1'
df = pd.read_csv(file_name)
 
# separate the data into the features and classification
X = df.drop(columns = ['Change_T10'])
y = df['Change_T10']

# implement the svm

model = svm.SVC(kernel='linear')

# use the first 1111 entries as the training data set
# we do not exlucde the 1111 entires from the testing set
model.fit(X[0:1111], y[0:1111])
y_predicted = model.predict(X)
 
# assses accuracy of model
prediction_array = (y_predicted == y)
score = sum(prediction_array)
accuracy = score/ y.shape[0]
print('The model has an accuracy of {}%'.format(round(accuracy*100,2)))

