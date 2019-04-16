rm(list=ls())
setwd("C:\\Users\\Low HH Family\\Desktop\\Cloud\\OD\\BCG 4.2\\BC3409 AI in Finance and Accounting\\Project")

# All 649 Variables
data1 <- read.csv("data.csv")

# Check that Upgraded and SuppCard are recognized as categorical in R.
summary(data1)

data1$Change_T10 <- factor(data1$Change_T10, levels=c(0,1), labels=c("No","Yes"))

#install.packages("rpart")
library(rpart)
#install.packages("rpart.plot")
library(rpart.plot)			# For Enhanced tree plots via PRP()

set.seed(2004)
options(digits = 3)

# default cp = 0.01. Set cp = 0 to guarantee no pruning in order to complete phrase 1: Grow tree to max.
# All 649 Variables
m1 <- rpart(Change_T10~., data = data1, method = 'class', cp = 0)

# Proper rpart plot from rpart.plot package.
#Better than plot() in rpart package.
prp(m1)
prp(m1, type=2, extra=104, nn=T, nn.box.col = 'light blue')

# Results of CART as Decision Rules
print(m1)

# Effects of Cost Complexity Pruning at important cp values.
printcp(m1, digits = 3)

# Plot CV error vs cp values
plotcp(m1)
## m1 tree is very small. why?

#Default is 30, only 30 data, therefore only one split


# Default minsplit = 30. Due to small sample size, 
# to reduce this value in order to get bigger tree.
# All 649 Variables
m2 <- rpart(Change_T10~., data = data1, method = 'class', control = rpart.control(minsplit = 2, cp = 0))

print(m2)
prp(m2, type=2, extra=104, nn=T, nn.box.col = 'light blue')
printcp(m2, digits = 3)
plotcp(m2)

summary(m2)

# Optimal CP = CP that result in lowest CV error. Too simplistic. ok for now.
# This statistical opinion can be overwritten by expert opinion, if any.
cp.opt <- m2$cptable[which.min(m2$cptable[,"xerror"]),"CP"]

# Prune the max tree m2 using a particular CP value (i.e. a specified penalty cost for model complexity)
m3 <- prune(m2, cp = cp.opt)
print(m3)
printcp(m3, digits = 5)

## --- Trainset Error & CV Error --------------------------
## Root node error: 13/30 = 0.433
## m3 trainset error = 0.231 * 0.433 = 0.1 = 10%
## m3 test set error = CV error = 0.692 * 0.433 = 0.3 = 30%


# Plot the final tree chosen.
prp(m3, type=2, extra=104, nn=T, fallen.leaves=T, branch.lty=3, nn.box.col = 'light blue', min.auto.cex = 0.7, nn.cex = 0.6, split.cex = 1.1, shadow.col="grey")
## By default, cases that satisfy the split condition go to the left child node; Other cases go to right child node.


m3$variable.importance
## Purchases is higher in importance than SuppCard in m3. Contributed more towards improving node purity.


# A new set of data (i.e. potential client list) can be used to generate m3 Tree Predictions, similar to Hotel exercise.
# Here, we illustrate CART model predictions on trainset data1.
predicted <- predict(m3, newdata = data1, type='class')
summary(predicted)

# Confusion Matrix can be constructed by applying model prediction on testset.
# Illustrated using trainset data1 as testset is not available.
table(data1$Change_T10, predicted)


summary(m3)

# ---------------------- END ---------------------------------

