###########################################################################
##This code is for EAS506 Final Project Model Fitting parts
##Created: Yuan Hui
##11/27/2018
###########################################################################
setwd("E:/Buffalo/Data_Science/Semester1/Data_Mining_One/project")

rm(list=ls())

library(ElemStatLearn)
library(glmnet)
library(geneplotter)
library(pls)
library(xgboost)
library(caret)
library(rpart)
library(gbm)
library(randomForest)
library(geneplotter)
library(ISLR)
library(neuralnet)
library(e1071)


###########################################################################
##Load data
###########################################################################
train <- read.csv("train_Yuan.csv")

train_x <- as.matrix(train[,-79])
train_y <- train[,79]

test_x <- as.matrix(read.csv("test_Yuan.csv"))
test_y <- read.csv("sample_submission.csv")

###########################################################################
##Ridge model
###########################################################################

ridge.mod <- glmnet(train_x,train_y,alpha=0)
plot(ridge.mod)
names(ridge.mod)

set.seed(123)
cv.out <- cv.glmnet(train_x,train_y,alpha=0)

x11()
plot(cv.out)


names(cv.out)
bestlam <- cv.out$lambda.min
bestlam  ###176411.9

ridge.pred <- predict(ridge.mod, s=bestlam, type="coefficient")
y_hat <- predict(ridge.mod, s=bestlam, newx=test_x,type="response")

test_error <- (sum(abs((y_hat - test_y)/test_y)))/nrow(test_y) 
test_error  ## 0.11

###########################################################################
##Lasso model
###########################################################################
lasso.mod <- glmnet(train_x,train_y,alpha=1)


x11()
plot(lasso.mod)
##coef(lasso.mod)
names(lasso.mod)

set.seed(123)
cv.out.lasso=cv.glmnet(train_x,train_y,alpha=1)
x11()
plot(cv.out.lasso)
bestlam=cv.out$lambda.min
bestlam   ##3155

lasso.pred <- predict(lasso.mod, s=bestlam, type="coefficients")
y_hat_lasso <- predict(lasso.mod, s=bestlam, newx=test_x,type="response")

test_error_lasso <- (sum(abs((y_hat_lasso - test_y)/test_y)))/nrow(test_y) 
test_error_lasso  ## 0.12

###########################################################################
##PCA model
###########################################################################

set.seed(123)

pcr.fit = pcr(SalePrice ~., data = train, scale = TRUE, validation = "none")
#summary(pcr.fit)
x11()
validationplot(pcr.fit, val.type = "RMSEP")
####Evaluate the performance of the model with i components in the pca regression fortest and trainning

testing_error_store <- c()

for (i in 1:78){
    pcr.pred.test = predict(pcr.fit, test_x, ncomp =i)
    test.error <- (sum(abs((pcr.pred.test - test_y)/test_y)))/nrow(test_y)
    testing_error_store <- c(testing_error_store, test.error)
}

testing_error_store
which.min(testing_error_store)##4
min(testing_error_store)  ##0.13

x11()
plot(testing_error_store)

###########################################################################
##XgBoost model
###########################################################################
xgb_params <- list(
  booster = 'gbtree',
  objective = 'reg:linear',
  colsample_bytree=1,
  eta=0.05,
  max_depth=3,
  min_child_weight=4,
  alpha=0.3,
  lambda=0.4,
  gamma=0, # less overfit
  subsample=1,
  seed=5,
  silent=TRUE)

dtrain <- xgb.DMatrix(as.matrix(train_x), label = train_y)
dtest <- xgb.DMatrix(as.matrix(test_x))


set.seed(12345)
bst <- xgb.train(xgb_params,dtrain, nrounds = 450)

y_pred.xgb <- predict(bst, dtest)

test_error_xgb <- (sum(abs((y_pred.xgb - test_y)/test_y)))/nrow(test_y) 
test_error_xgb  ##0.1

mat <- xgb.importance (feature_names = colnames(train_x),model = bst)
xgb.plot.importance (importance_matrix = mat[1:30],main="Important Features",xlab= "Frequency", col='blue')

###########################################################################
##Change the problem into a binary classification problem
###########################################################################
mean(train_y)  ##180932
##Recode sales as a binary response
High <- ifelse(train_y<=180932,"No","Yes")
train_new <- data.frame(train_x,High)
High_test <- ifelse(test_y<=180932,0,1)
test_new=data.frame(test_x,High_test)

###########################################################################
##Random Forest
###########################################################################

rf.fit <- randomForest(High~.,data=train_new, n.tree =10000)
names(rf.fit)

x11()
varImpPlot(rf.fit)
importance(rf.fit)
y_hat <- predict(rf.fit, newdata=test_x, type="response") 
y_hat=as.numeric(y_hat)-1  ##No is 0, Yes is 1
misclass_rf <- sum(abs(High_test-y_hat))/length(y_hat)
misclass_rf ##0.4

###########################################################################
##SVM_Linear
###########################################################################
##SVM with a linear kernal

tune.model <- tune(svm, High~., data=train_new, kernal='linear',
ranges=list(cost=c(0.001, 0.01, 1, 5, 10,100)))  #cost=1

tune.model

summary(tune.model)
bestmod <- tune.model$best.model
bestmod

##predict the test data

y_hat <- predict(bestmod, newdata=test_x)
y_true <- ifelse(test_y<=180932,"No","Yes")
error_lin <- length(which(y_hat!=y_true))/length(y_true)
error_lin  ##0.43

##Confusion table
table(predict = y_hat,truth=y_true)




