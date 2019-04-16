# http://dataaspirant.com/2017/01/19/support-vector-machine-classifier-implementation-r-caret-package/
rm(list=ls())
setwd("C:\\Users\\Low HH Family\\Desktop\\Cloud\\OD\\BCG 4.2\\BC3409 AI in Finance and Accounting\\Project")
# Read data
data = read.csv('data.csv')
summary(data)
str(data)
head(data)
anyNA(data)

library(caret)

set.seed(3061)
intrain <- createDataPartition(y = data$Change_T10, p= 0.7, list = FALSE)
training <- data[intrain,]
testing <- data[-intrain,]
dim(training)
dim(testing)

training[["Change_T10"]] = factor(training[["Change_T10"]])

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3233)

############# Linear SVM model ############# 
svm_Linear <- train(Change_T10 ~., data = training, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

svm_Linear
plot(svm_linear)

test_pred <- predict(svm_Linear, newdata = testing)
test_pred

confusionMatrix(test_pred, factor(testing$Change_T10))




############# Grid-Linear SVM model #############
grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
set.seed(3233)
svm_Linear_Grid <- train(Change_T10 ~., data = training, method = "svmLinear",
                         trControl=trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)

svm_Linear_Grid
plot(svm_Linear_Grid)

test_pred_grid <- predict(svm_Linear_Grid, newdata = testing)
test_pred_grid

confusionMatrix(test_pred_grid, factor(testing$Change_T10))




############# Radial SVM model #############
set.seed(3233)
svm_Radial <- train(Change_T10 ~., data = training, method = "svmRadial",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
svm_Radial
plot(svm_Radial)

test_pred_radial <- predict(svm_Radial, newdata = testing)
test_pred_radial

confusionMatrix(test_pred_radial, factor(testing$Change_T10))



'''
############# Grid-Radial SVM model #############
grid_radial <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 0.03, 0.04,
                                     0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.25, 0.5, 0.75,0.9),
                           C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
                                 1, 1.5, 2,5))
set.seed(3233)
svm_Radial_Grid <- train(Change_T10 ~., data = training, method = "svmRadial",
                         trControl=trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid_radial,
                         tuneLength = 10)

svm_Radial_Grid
plot(svm_Radial_Grid)

test_pred_Radial_Grid <- predict(svm_Radial_Grid, newdata = testing)
confusionMatrix(test_pred_Radial_Grid, factor(testing$Change_T10))
'''