import sklearn
from sklearn import svm
import numpy as np
from csv import reader 

EPOCHS = 100
w = 0
r = 1
iteration = [0] * 30

def read_file(fileName):
    dataset = list()
    with open (fileName, 'r', newline = '', encoding = 'utf-8') as file:
        values = reader(file)
        for row in values:
            if not row:
                continue
            dataset.append(row)
    return dataset

test = read_file('data_svm_train.csv')
target = read_file('data_svm_test.csv')

for i in range(len(iteration)):
    r += 10
    if (r + 10) > 1597:
        r = w
        w += 300

    traindataset = [test[i] for i in range(r, r + 10)]
    train_target = [target[i] for i in range(r, r + 10)]
    model = svm.SVC(kernel="rbf", decision_function_shape='ovo')
    model.fit(traindataset, train_target)

    # Model Testing
    count = 0
    total = 0
    for j in range(len(test)):
        total += 1
        temp = model.predict([test[j]])
        if temp == target[j]:
            count += 1
    accuracy = count * 100 / total
    print('Accuracy: %s' % accuracy)
    iteration[i] = accuracy

mean = sum(iteration)/len(iteration)
print("Number of Epochs: {}".format(EPOCHS))
print("Mean_Accuracy: {}".format(mean))
print("Standard_Deviation: {}".format(np.std(iteration, 0)))
