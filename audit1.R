#Load libraries
library(mlbench)
library(caret)
library(dplyr)

#Load data. The data was obtained from https://archive.ics.uci.edu/ml/datasets/Audit+Data
dataset <- audit_risk

#The data had just one missing value which needed to be discarded
dataset <- na.omit(dataset)

#Of the 26 predictors, 2 are near zero variance predictors-location_id and detection_risk which are eliminated
dataset <- select(dataset, -2, -25)

#The target variable is converted to categorical
dataset$Risk = as.factor(dataset$Risk)

#10 fold cross validation with 3 repeats is used
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7

#The evaluation metric used is accuracy
metric <- "Accuracy"

#The data is centered and scaled for preprocessing
preProcess=c("center", "scale")

#Algorithm spot check

# Linear Discriminant Analysis
set.seed(seed)
fit.lda <- train(Risk~., data=dataset, method="lda", metric=metric, preProc=c("center", "scale"), trControl=control)
# Logistic Regression
set.seed(seed)
fit.glm <- train(Risk~., data=dataset, method="glm", metric=metric, trControl=control)
# kNN
set.seed(seed)
fit.knn <- train(Risk~., data=dataset, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# SVM Radial
set.seed(seed)
fit.svmRadial <- train(Risk~., data=dataset, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
# Naive Bayes
set.seed(seed)
fit.nb <- train(Risk~., data=dataset, method="nb", metric=metric, trControl=control)
# CART
set.seed(seed)
fit.cart <- train(Risk~., data=dataset, method="rpart", metric=metric, trControl=control)
## Random Forest
set.seed(seed)
fit.rf <- train(Risk~., data=dataset, method="rf", metric=metric, trControl=control)

#Evaluation of algorithms
results <- resamples(list(lda=fit.lda, logistic=fit.glm, knn=fit.knn, nb=fit.nb,svm=fit.svmRadial, cart=fit.cart, rf=fit.rf))

# Table comparison
summary(results)

# boxplot comparison
bwplot(results)

# Dot-plot comparison
dotplot(results)

#Tree based algorithms like CART and Random Forest give 100% accuracy because they suffer from overfitting. I would like to investigate them further after implementing pruning.
#Non linear algorithms like KNN, Naive Bayes and SVM perform slightly better than the linear ones like LDA and Logistic regression.
#I would like to further evaluate these alogorithms for other metrics like speed and precision.                  