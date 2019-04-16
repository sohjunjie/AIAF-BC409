rm(list=ls())

library(randomForest)

setwd("C:\\Users\\Low HH Family\\Desktop\\Cloud\\OD\\BCG 4.2\\BC3409 AI in Finance and Accounting\\Project")
# Transforming the dependent variable to a factor
data <- read.csv("data.csv")
data$Change_T10 = as.factor(data$Change_T10)

# Data Exploration
head(data)
str(data)
summary(data)

# Splitting into training and testing
index <- sample(nrow(data), 0.731*nrow(data), replace = FALSE)
train <- data[index,]
test <- data[-index,]
summary(train)
summary(test)

# Creating the Random Forest Model
model <- randomForest(Change_T10~ ., data = train, importance = TRUE)
model

# Using For loop to identify the right mtry for model
a=c()
i=5
for (i in 3:8) {
  model2 <- randomForest(Change_T10~ ., data = train, ntree = 500, 
                         mtry = i, importance = TRUE)
  predValid <- predict(model2, test, type = "class")
  a[i-2] = mean(predValid == test$Change_T10)
}
a

# Fine tuning parameters of Random Forest model
model1 <- randomForest(Change_T10~ ., data = train, 
                       ntree = 500, mtry = 7, importance = TRUE)
model1

# Predicting on train set
predTrain <- predict(model1, train, type = "class")
# Checking classification accuracy
table(predTrain, train$Change_T10)

# Predicting on test set
predValid <- predict(model1, test, type = "class")
# Checking classification accuracy
mean(predValid == test$Change_T10)                    
table(predValid,test$Change_T10)

# To check important variables
importance(model1)        
# varImpPlot(model1)