rm(list=ls())
setwd("C:\\Users\\Low HH Family\\Desktop\\Cloud\\OD\\BCG 4.2\\BC3409 AI in Finance and Accounting\\Project")
# Read data
data = read.csv('data.csv')
summary(data)
str(data)
anyNA(data)

library(caret)

set.seed(6310)
intrain <- createDataPartition(y = data$Change_T10, p= 0.7, list = FALSE)
training <- data[intrain,]
testing <- data[-intrain,]

dim(training)
dim(testing)

training[["Change_T10"]] = factor(training[["Change_T10"]])

#Training the KNN Model
trctrl <- trainControl(method = "repeatedcv", 
                       number = 10, repeats = 3)
set.seed(1360)
knn_fit <- train(Change_T10 ~., data = training, method = "knn",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)

# View Model
knn_fit
plot(knn_fit)

# Test Set Prediction
test_pred <- predict(knn_fit, newdata = testing)
test_pred

# Confusion Matrix
confusionMatrix(test_pred, factor(testing$Change_T10))
