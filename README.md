# BC3409 - AI in Accounting and Finance

## Project Title
Predicting Financial Market Movement using AI

## Project Description
The aim of the course project is to apply AI models into accounting and finance.
Our team decided to work on the prediction of NASDAQ stock index using a variety
of [AI models](#implemented-ai-models).

## Project Members
1. Ernest Lim ([ernestlwt](https://github.com/ernestlwt))
2. Koh Kian Woon ([kianwoon123](https://github.com/kianwoon123))
3. Terson Tan ([tersontan](https://github.com/tersontan))
4. Liu Shanyi ([Hushini631](https://github.com/Hushini631))
5. Soh Jun Jie ([sohjunjie](https://github.com/sohjunjie))

## Package requirements
Please install the required python packages with the following commands.
```
$ pip install -r requirements.txt
```

## Implemented AI models
#### 1. Decision Tree, Random Forest, KNN, SVM
Basic models for predicting stock movement and price are implemented in `R` in the
[folder](_nasdaq_stock_price/basic_model_R).
```
1. decision tree.R         # decision tree model
2. knn.R                   # K-Nearest Neighbour model
3. LR.R                    # Linear Regression model
4. svm.R                   # Support Vector Model
```

#### 2. RNN, GRU, LSTM
This model uses the stock price information from NASDAQ and 9 other top NASDAQ companies to predict the 9 companies stock price trends.
TA-lib was also used to generate additional variables from those information.

Starting from the project root directory, you can test and execute the model using the following commands.
```
$ cd _nasdaq_stock_price
$ python rnn_top9.py
```
You can also can change the model parameters easily at the area just below the import statements

#### 3. GRU + News
This model uses time series news data and market prices to predict NASDAQ price trends.
Starting from the project root directory, you can test and execute the model using the following commands.
```
$ cd _nasdaq_stock_price
$ python gru.py
```
