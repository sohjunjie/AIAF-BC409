rm(list=ls())
setwd("C:\\Users\\Low HH Family\\Desktop\\Cloud\\OD\\BCG 4.2\\BC3409 AI in Finance and Accounting\\Project")

data <-read.csv("data_LR.csv")
head(data)
plot(data)
model <- Close~.
result <-lm(model,data)
summary(result)

r1 <- c(0,max(data$Close))
rrange <- range(c(r1,r1))
plot (result$fitted.values,data$Close,xlim=rrange,ylim=rrange)
lines(r1,r1)
lines(lowess(result$fitted.values,data$Close,f=0.8),col=c("red"))
sum(result$fitted.values)
sum(data$Close)
cor(data$Close,result$fitted.values)
epsilon1 <- rnorm(10)
epsilon2 <- rnorm(10)
fit_y <- 10 +4*epsilon1
y <- fit_y +1.5*epsilon2
xrange <-range(fit_y)
yrange <- range(y)
rrange <- range(c(xrange,yrange))
plot(fit_y,y,xlim=xrange,ylim=rrange)
lines(xrange,yrange)
residual <- y-fit_y
yrange <- range(residual)*2
plot(y,residual,xlim=xrange,ylim=yrange)
lines(yrange,c(0,0))
