rm(list = ls())    # reset / remove all objects
cat("\014") # clear console
library(dplyr)
library(tidyverse)
library(data.table)
library(glmnet)
library(randomForest)
library(ISLR)
library(MASS)
library(ggplot2)
library(class)
library(coefplot)
library(base)


### CLEARING ORIGINAL DATASET INTO A DATAFRAME "DF.csv"
### ORIGINAL DATASET IS AVAILABLE THROUGH LINK IN README.txt FILE

# bottle <- read_csv("bottle.csv")

## remove columns containing only NA's

# bottle2 <- bottle[, colSums(is.na(bottle)) != nrow(bottle)]

# DF <- data.frame(bottle2)

## replace missing observations with column mean

# for(i in 1:ncol(DF)){
 # DF[is.na(DF[,i]), i] <- mean(DF[,i], na.rm = TRUE)
# }

## drop first columns which are ID's

# DF <- DF[-c(1:5)]

## reduce number of rows

# DF <- DF[sample(1:nrow(DF), 0.0015*nrow(DF)),]
# write.csv(DF,'DF.csv', row.names=F)


DF <- read_csv("DF.csv")

# log depdendent variable
y = log(DF$T_degC + 1)

# data info
mean(y)
median(y)
sd(y)
hist(DF$T_degC)
hist(y)

# dimensions of the data
DF$T_degC <- NULL
n = dim(DF)[1] #n = 1297
p = dim(DF)[2] #p = 36

# convert to X matrix
regressors  = DF
mat = data.matrix(regressors)


# 100 SAMPLES

# 80/20 split and repetition count
set.seed(2)
n.train = floor(0.8 * n)
n.test = n - n.train
run_times = 100

# lasso
Rsq.train.lasso = rep(0,run_times)
Rsq.test.lasso = rep(0,run_times) 

# elastic net
Rsq.train.en = rep(0,run_times)
Rsq.test.en = rep(0,run_times) 

# ridge
Rsq.train.ridge = rep(0,run_times)
Rsq.test.ridge = rep(0,run_times)

# randomForest
Rsq.train.rf = rep(0,run_times)
Rsq.test.rf = rep(0,run_times)

# run 100 times

for (i in c(1:run_times)) {
  
  # randomly split the data into train and test
  shuffled_indexes = sample(n)
  train = shuffled_indexes[1:n.train]
  test = shuffled_indexes[(1+n.train):n]
  
  X.train = mat[train,]
  y.train = y[train]
  X.test = mat[test,]
  y.test = y[test]
  
  # lasso
  lasso.cv = cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  lasso.fit = glmnet(X.train, y.train, alpha = 1, lambda = lasso.cv$lambda.min)
  y.train.hat = predict(lasso.fit, newx = X.train, type = "response") 
  y.test.hat = predict(lasso.fit, newx = X.test, type = "response") 
  Rsq.train.lasso[i] = 1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.lasso[i] = 1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  # elastic net 
  a = 0.5
  en.cv = cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  en.fit = glmnet(X.train, y.train, alpha = a, lambda = en.cv$lambda.min)
  y.train.hat = predict(en.fit, newx = X.train, type = "response") 
  y.test.hat = predict(en.fit, newx = X.test, type = "response") 
  Rsq.train.en[i] = 1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.en[i] = 1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  # ridge
  ridge.cv = cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  ridge.fit = glmnet(X.train, y.train, alpha = 0, lambda = ridge.cv$lambda.min)
  y.train.hat = predict(ridge.fit, newx = X.train, type = "response") 
  y.test.hat = predict(ridge.fit, newx = X.test, type = "response") 
  Rsq.train.ridge[i] = 1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.ridge[i] = 1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  # random forest
  rf.fit = randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.train.hat = predict(rf.fit, X.train)
  y.test.hat = predict(rf.fit, X.test)
  Rsq.train.rf[i] = 1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.rf[i] = 1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
}

# R-SQUARED BOXPLOTS 

R2 = data.frame(c(rep("Train", 4*run_times), rep("Test", 4*run_times)), 
                    c(rep("Lasso",run_times),rep("Elastic-net",run_times), 
                      rep("Ridge",run_times),rep("Random Forest",run_times), 
                      rep("Lasso",run_times),rep("Elastic-net",run_times), 
                      rep("Ridge",run_times),rep("Random Forest",run_times)), 
                    c(Rsq.train.lasso, Rsq.train.en, Rsq.train.ridge, Rsq.train.rf, 
                      Rsq.test.lasso, Rsq.test.en, Rsq.test.ridge, Rsq.test.rf))

colnames(R2) =  c("Type", "Method", "Rsqr")
R2$Method = factor(R2$Method, levels=c("Lasso", "Elastic-net", "Ridge", "Random Forest"))
R2$Type = factor(R2$Type, levels=c("Train", "Test"))

ggplot(R2,aes(x = Method, y = Rsqr, fill = Method)) + geom_boxplot() + facet_wrap(~Type, ncol = 2) +
  theme(axis.text.x = element_text(hjust = 1,vjust = 0.5, size = 30/.pt, angle = 90), plot.title=element_text(hjust = 0.5)) + 
  ggtitle("R2 of Train and Test")


# CV CURVES FOR ONE SAMPLE AND TIME

# Lasso 

ptm=proc.time()
lasso.cv = cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
ptm = proc.time() - ptm
time_lasso  =ptm["elapsed"]
cat(sprintf("Lasso: %0.3f(sec):",time_lasso))

# Plot CV Curve
plot(lasso.cv)+title("10-fold CV curve for Lasso", line = 2.5)


# Elastic-net

a = 0.5
ptm = proc.time()
en.cv =cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
ptm  = proc.time() - ptm
time_en=   ptm["elapsed"]
cat(sprintf("Elastic-net: %0.3f(sec):",time_en))

# Plot CV Curve
plot(en.cv)+title("10-fold CV curve for Elastic-net", line = 2.5)


# Ridge

ptm=proc.time()
ridge.cv=cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
ptm=proc.time() - ptm
time_ridge= ptm["elapsed"]
cat(sprintf("Ridge: %0.3f(sec):",time_ridge))

# Plot CV Curve
plot(ridge.cv)+title("10-fold CV curve for Ridge", line = 2.5)


## RESIDUAL BOXPLOTS

# Lasso

lasso.fit = glmnet(X.train, y.train, alpha = 1, lambda = lasso.cv$lambda.min)

y.train.hat.lasso = predict(lasso.fit, newx = X.train, type = "response") 
y.test.hat.lasso = predict(lasso.fit, newx = X.test, type = "response") 

Rsq.train_lasso = 1 - mean((y.train - y.train.hat.lasso)^2)/mean((y - mean(y))^2)  
Rsq.test_lasso= 1 - mean((y.test - y.test.hat.lasso)^2)/mean((y - mean(y))^2)

y.train.hat.lasso = as.vector(y.train.hat.lasso)
y.test.hat.lasso = as.vector(y.test.hat.lasso)

residual.lasso=data.table(c(rep("Train", n.train),rep("Test", n.test)),   
                          c(1:n), c(y.train.hat.lasso-y.train, y.test.hat.lasso-y.test))
colnames(residual.lasso) = c("Type", "Data", "Residual")


# Elastic-net

en.fit= glmnet(X.train, y.train, alpha = a, lambda = en.cv$lambda.min)

y.train.hat.en=predict(en.fit, newx = X.train, type = "response") 
y.test.hat.en=predict(en.fit, newx = X.test, type = "response") 

Rsq.train_en=1-mean((y.train - y.train.hat.en)^2)/mean((y - mean(y))^2)  
Rsq.test_en=1-mean((y.test - y.test.hat.en)^2)/mean((y - mean(y))^2)

y.train.hat.en =as.vector(y.train.hat.en)
y.test.hat.en =as.vector(y.test.hat.en)

residual.en = data.table(c(rep("Train", n.train),rep("Test", n.test)),   
                         c(1:n), c(y.train.hat.en-y.train, y.test.hat.en-y.test))
colnames(residual.en) =  c("Type", "Data", "Residual")


# Ridge

ridge.fit=glmnet(X.train, y.train, alpha = 0, lambda = ridge.cv$lambda.min)

y.train.hat.ridge = predict(ridge.fit, newx = X.train, type = "response") 
y.test.hat.ridge = predict(ridge.fit, newx = X.test, type = "response")

Rsq.train_ridge=1 - mean((y.train - y.train.hat.ridge)^2)/mean((y - mean(y))^2)  
Rsq.test_ridge=1 - mean((y.test - y.test.hat.ridge)^2)/mean((y - mean(y))^2)

y.train.hat.ridge =as.vector(y.train.hat.ridge)
y.test.hat.ridge =as.vector(y.test.hat.ridge)

residual.ridge = data.table(c(rep("Train", n.train),rep("Test", n.test)),  
                            c(1:n), c(y.train.hat.ridge-y.train, y.test.hat.ridge-y.test))
colnames(residual.ridge) =  c("Type", "Data", "Residual")


# Random Forest

rf.fit=randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)

y.train.hat.rf=predict(rf.fit, X.train)
y.test.hat.rf= predict(rf.fit, X.test)

Rsq.train_rf= 1 - mean((y.train - y.train.hat.rf)^2)/mean((y - mean(y))^2)  
Rsq.test_rf= 1 - mean((y.test - y.test.hat.rf)^2)/mean((y - mean(y))^2)

y.train.hat.rf =as.vector(y.train.hat.rf)
y.test.hat.rf =as.vector(y.test.hat.rf)

residual.rf = data.table(c(rep("Train", n.train),rep("Test", n.test)),  
                         c(1:n), c(y.train.hat.rf-y.train, y.test.hat.rf-y.test))
colnames(residual.rf) =  c("Type", "Data", "Residual")


# Consolidate all residuals in a long format data.table
residual.dt = data.frame(c(rep("Lasso",n), rep("Elastic-net",n), rep("Ridge",n), rep("Random Forest",n)),
                         rbind(residual.lasso, residual.en, residual.ridge, residual.rf))

colnames(residual.dt) = c("Method", "Type", "Data", "Residual")
residual.dt$Method = factor(residual.dt$Method, levels = c("Lasso", "Elastic-net", "Ridge", "Random Forest"))
residual.dt$Type = factor(residual.dt$Type, levels = c("Train", "Test"))

# Plot boxplot using ggplot
ggplot(residual.dt, aes(x = Method, y = Residual, fill = Method)) + geom_boxplot() + facet_wrap(~Type, ncol = 2) + 
  theme(axis.text.x = element_text(angle = 90,hjust = 1,vjust = 0.5, size = 30/.pt)) +
  ggtitle("Train and Test Residuals") + theme(plot.title = element_text(hjust = .5,size = 40/.pt))

## FIT ALL DATA

# Lasso

ptm=proc.time()
lasso.cv = cv.glmnet(mat, y, alpha = 1, nfolds = 10)
lasso.fit = glmnet(mat, y, alpha = 1, lambda = lasso.cv$lambda.min)
ptm = proc.time() - ptm
time_lasso = ptm["elapsed"]
cat(sprintf("Run Time for Lasso: %0.3f(sec):",time_lasso))

y.hat.lasso = predict(lasso.fit, newx = mat, type = "response") 
Rsq_lasso = 1 - mean((y - y.hat.lasso)^2) / mean((y - mean(y))^2)

## Elastic-net

a = 0.5
ptm = proc.time()
en.cv = cv.glmnet(mat, y, alpha = a, nfolds = 10)
en.fit = glmnet(mat, y, alpha = a, lambda = en.cv$lambda.min)
ptm= proc.time() - ptm
time_en = ptm["elapsed"]
cat(sprintf("Run Time for elastic-net: %0.3f(sec):",time_en))

y.hat.en = predict(en.fit, newx = mat, type = "response") 
Rsq_en = 1 - mean((y - y.hat.en)^2) / mean((y - mean(y))^2)  

## Ridge

ptm=proc.time()
ridge.cv = cv.glmnet(mat, y, alpha = 0, nfolds = 10)
ridge.fit = glmnet(mat, y, alpha = 0, lambda = ridge.cv$lambda.min)
ptm=proc.time() - ptm
time_ridge= ptm["elapsed"]
cat(sprintf("Run Time for ridge: %0.3f(sec):",time_ridge))

y.hat.ridge = predict(ridge.fit, newx = mat, type = "response") 
Rsq_ridge = 1 - mean((y - y.hat.ridge)^2) / mean((y - mean(y))^2)  

## Random Forest

ptm=proc.time()
rf.fit=randomForest(mat, y, mtry = sqrt(p), importance = TRUE)
ptm = proc.time() - ptm
time_rf = ptm["elapsed"]
cat(sprintf("Run Time for Random Forest: %0.3f(sec):",time_rf))

y.hat.rf=predict(rf.fit, mat)
Rsq_rf= 1-mean((y - y.hat.rf)^2)/mean((y - mean(y))^2)  

# PERFORMANCE VS TIME

# for the 100 repetitions
lasso.Rsq.ci = t.test(Rsq.test.lasso, conf.level = 0.9)
en.Rsq.ci = t.test(Rsq.test.en, conf.level = 0.9)
ridge.Rsq.ci = t.test(Rsq.test.ridge, conf.level = 0.9)
rf.Rsq.ci = t.test(Rsq.test.rf, conf.level = 0.9)
lasso.Rsq.ci$conf.int[1:2]
en.Rsq.ci$conf.int[1:2]
ridge.Rsq.ci$conf.int[1:2]
rf.Rsq.ci$conf.int[1:2]

# time and performance in data.table
time_perf.dt = data.table(rep(c("Lasso", "Elastic-net", "Ridge", "Random Forest"),2) ,
                          rep(c(time_lasso, time_en, time_ridge, time_rf),2),
                          rep(c("Lower Bound", "Upper Bound"),4),
                          c(lasso.Rsq.ci$conf.int[1], en.Rsq.ci$conf.int[1], ridge.Rsq.ci$conf.int[1], rf.Rsq.ci$conf.int[1],
                            lasso.Rsq.ci$conf.int[2], en.Rsq.ci$conf.int[2], ridge.Rsq.ci$conf.int[2], rf.Rsq.ci$conf.int[2]))

colnames(time_perf.dt) = c("Method", "Time", "Bound Type", "Rsqr")

# Plot Time against 90% Confidence Intervals for R-squared
ggplot(time_perf.dt, aes(x = Time, y = Rsqr, color = Method)) + geom_point() + ylab("R2") + xlab("Time (sec)") + 
  ggtitle("Trade-off") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

# VARIABLE ANALYSIS 

# Estimated Coefficients for Lasso, Elastic-net, and Ridge
lasso_beta = data.table(as.character(c(1:p)), as.vector(lasso.fit$beta))
en_beta = data.table(as.character(c(1:p)), as.vector(en.fit$beta))
ridge_beta = data.table(as.character(c(1:p)), as.vector(ridge.fit$beta))

# Variable Importance for Random Forest
rf_importance = data.table(as.character(c(1:p)), as.vector(rf.fit$importance[,1]))

# Rename columns for uniformity 
colnames(lasso_beta) = c("param", "value")
colnames(en_beta) = c("param", "value")
colnames(ridge_beta) = c("param", "value")
colnames(rf_importance) = c("param", "value")

# Ordering all methods by Elastic-net's estimated coefficients
en_beta$param = factor(en_beta$param, levels = en_beta$param[order(en_beta$value, decreasing= TRUE)])
rf_importance$param = factor(rf_importance$param, levels = en_beta$param[order(en_beta$value, decreasing= TRUE)])
lasso_beta$param = factor(lasso_beta$param, levels = en_beta$param[order(en_beta$value, decreasing= TRUE)])
ridge_beta$param = factor(ridge_beta$param, levels = en_beta$param[order(en_beta$value, decreasing= TRUE)])

# creating a data.table for residuals
rf_importance$Method = "Random Forest" 
lasso_beta$Method = "Lasso" 
en_beta$Method = "Elastic-net" 
ridge_beta$Method = "Ridge" 

par <- data.table(rbind(en_beta, lasso_beta, ridge_beta, rf_importance))

colnames(par) = c("Parameter" ,"Value", "Method")

par$Method = factor(par$Method, levels = c("Elastic-net","Lasso", "Ridge", "Random Forest"))

ggplot(par,aes(x = Parameter, y = Value, fill = Method)) + geom_col() +
  facet_wrap(~Method, nrow = 4, scales= "free_y") +
  theme(axis.text.x = element_text(angle = 90,hjust = 1,vjust = 0.5, size = 30/.pt)) +
  ggtitle("Importance of the parameters ") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))
