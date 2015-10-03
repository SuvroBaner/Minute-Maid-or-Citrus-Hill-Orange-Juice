#################### Orange Juice Data ##############################

# Dataset : OJ

# Description: The data set contains 1070 purchases where the customer either purchased Citrus Hill or Minute Maid
# orange juice. A number of characterstics of the customer and product are recorded (18 variables)

# Citrus Hill was a brand of orange juice introduced by Procter & Gamble in the United States market in 1982,
# later used for other fruit juices and beverages.

# The Minute Maid company is owned by The Coca-Cola Company, and is the world's largest marketer of fruit juices and drinks.

# Data set:

# Purchase: A factor with levels CH and MM indicating whether the customer purchased Citrus Hill or Minute Maid Orange Juice
# WeekofPurchase: Week of purchase
# StoreID: Store ID
# PriceCH: Price charged for CH
# PriceMM: Price charged for MM
# DiscCH: Discount offered for CH
# DiscMM: Discount offered for MM
# SpecialCH: Indicator of special on CH
# SpecialMM: Indicator of special on MM
# LoyalCH: Customer brand loyalty for CH
# SalePriceMM: Sale price for MM
# SalePriceCH: Sale price for CH
# PriceDiff: Sale price of MM less sale price of CH
# Store7: A factor with levels No and Yes indicating whether the sale is at Store 7
# PctDiscMM: Percentage discount for MM
# PctDiscCH: Percentage discount for CH
# ListPriceDiff: List price of MM less list price of CH
# STORE: Which of 5 possible stores the sale occured at

install.packages("ISLR") # to access the data set.
install.packages("tree") # tree methods
install.packages("randomForest")  # for bagging and random forest
install.packages("e1071") # svm

library(ISLR)
library(tree)
library(randomForest)
library(e1071)

dim(OJ)
fix(OJ)
summary(OJ)
names(OJ)
attach(OJ)

##### Setting up the data ######

set.seed(1)
train = sample(nrow(OJ), 800)
OJ.train = OJ[train, ]
OJ.test = OJ[-train, ]

#### Fitting a tree to the training data ####

tree.OJ = tree(Purchase ~ . , data = OJ, subset = train)
summary(tree.OJ)

#Variables actually used in tree construction:
#[1] "LoyalCH"       "PriceDiff"     "SpecialCH"     "ListPriceDiff"
#Number of terminal nodes:  8 
#Residual mean deviance:  0.7305 = 578.6 / 792 
#Misclassification error rate: 0.165 = 132 / 800 

# The training error rate (misclassification rate) is 16.5 %

tree.OJ

# Lets pick on of the nodes-
# 10) PriceDiff < 0.195 83   91.66 MM ( 0.24096 0.75904 )
# The split criterion is  PriceDiff < 0.195
# There are 83 observations in the branch.
# The deviance is 91.66 % in this branch.
# The overall prediction of this branch is purchase of MM (Minute Maid)
# About 76 % have a purchase of MM and 24 % is for CH

plot(tree.OJ)
text(tree.OJ, pretty = 0)

# LoyaltyCH is the most important variable of the tree, infact the top three nodes are LoyaltyCH.

# Now let's predict the response on the test data.

OJ.pred = predict(tree.OJ, OJ.test, type = "class")
table(OJ.test$Purchase, OJ.pred)
mean(OJ.test$Purchase != OJ.pred)

# The test error rate is 22.6 %

###### Now let's use the cross-validation approach on the training set to determine the optimal tree size #######

cv.OJ = cv.tree(tree.OJ, FUN = prune.tree)  # guided by deviance 
names(cv.OJ)

plot(cv.OJ$size, cv.OJ$dev, type = "b", xlab = "Tree Size", ylab = "Deviance")
# The optimal tree size is 7

prune.OJ = prune.tree(tree.OJ, best = 7)

summary(prune.OJ)

# The misclassification training error rate is 16.5 % exactly the same as the unpruned tree.

OJ.prun.pred = predict(prune.OJ, OJ.test, type = "class")
table(OJ.test$Purchase, OJ.prun.pred)
mean(OJ.test$Purchase != OJ.prun.pred)

### The test error rate of the prunned tree is 22.6 % which is exactly same as the test error rate of the unpruned tree.

################# Now let's do bagging ###########

bag.OJ = randomForest(Purchase ~ . , data = OJ.train, mtry = 17, ntree = 500, importance = TRUE)
bag.pred = predict(bag.OJ, OJ.test, type = "class")
mean(OJ.test$Purchase != bag.pred)

# The test error rate using bagging is lower and it is 20.3%

########### Now let's do Random Forest ##############

rf.OJ = randomForest(Purchase ~ . , data = OJ.train, mtry = 4, ntree = 500, importance = TRUE)  # mtry = sqrt(no. of predictors)
rf.pred = predict(rf.OJ, OJ.test, type = "class")
mean(rf.pred != OJ.test$Purchase)

# The test error rate using random forest is also 20.3 %

################# SVM ######################

set.seed(9004)
train = sample(dim(OJ)[1], 800)
OJ.train = OJ[train, ]
OJ.test = OJ[-train, ]

# Fit a Support Vector Classifier to the training data using cost = 0.01

svm.linear = svm(Purchase ~ ., data = OJ.train, kernel = "linear", cost = 0.01)
summary(svm.linear)

# Support vector classifier creates 432 support vectors out of 800 training points. 
# Out of these, 217 belong to level CH and remaining 215 belong to level MM.

train.pred = predict(svm.linear, OJ.train)
table(OJ.train$Purchase, train.pred)
mean(OJ.train$Purchase != train.pred)

test.pred = predict(svm.linear, OJ.test)
table(OJ.test$Purchase, test.pred)
mean(OJ.test$Purchase != test.pred)


# The training error rate is 16.9% and test error rate is about 17.8%.

### Use the crossvalidation (tune()) to select an optimal cost. Consider value range from .01 to 10.

set.seed(15)

tune.out = tune(svm, Purchase ~ ., data = OJ.train, kernel = "linear", ranges = list(cost = 10^seq(-2, 
                                                                                                   1, by = 0.25)))
summary(tune.out)

# Tuning shows that optimal cost is 0.316  (using 10-fold cross validation)

svm.linear = svm(Purchase ~ ., kernel = "linear", data = OJ.train, cost = tune.out$best.parameters$cost)
train.pred = predict(svm.linear, OJ.train)
table(OJ.train$Purchase, train.pred)
mean(OJ.train$Purchase != train.pred)

test.pred = predict(svm.linear, OJ.test)
table(OJ.test$Purchase, test.pred)
mean(OJ.test$Purchase != test.pred)

# The training error decreases to 16% but test error slightly increases to 18.1% by using best cost.

### Fit an SVM with radial kernel.

set.seed(410)
svm.radial = svm(Purchase ~ ., data = OJ.train, kernel = "radial")
summary(svm.radial)

train.pred = predict(svm.radial, OJ.train)
table(OJ.train$Purchase, train.pred)
mean(OJ.train$Purchase != train.pred)

test.pred = predict(svm.radial, OJ.test)
table(OJ.test$Purchase, test.pred)
mean(OJ.test$Purchase != test.pred)

# The radial basis kernel with default gamma creates 367 support vectors, out of which, 184 belong to level CH and remaining 183 belong to level MM.
# The classifier has a training error of 16% and a test error of 15.6% which is a slight improvement over linear kernel. We now use cross validation to find optimal gamma.

set.seed(755)
tune.out = tune(svm, Purchase ~ ., data = OJ.train, kernel = "radial", ranges = list(cost = 10^seq(-2, 
                                                                                                   1, by = 0.25)))
summary(tune.out)

svm.radial = svm(Purchase ~ ., data = OJ.train, kernel = "radial", cost = tune.out$best.parameters$cost)
train.pred = predict(svm.radial, OJ.train)
table(OJ.train$Purchase, train.pred)
mean(OJ.train$Purchase != train.pred)

test.pred = predict(svm.radial, OJ.test)
table(OJ.test$Purchase, test.pred)
mean(OJ.test$Purchase != test.pred)

# Tuning slightly decreases training error to 14.6% and slightly increases test error to 15.9% 
# which is still better than linear kernel.

set.seed(8112)
svm.poly = svm(Purchase ~ ., data = OJ.train, kernel = "poly", degree = 2)
summary(svm.poly)

train.pred = predict(svm.poly, OJ.train)
table(OJ.train$Purchase, train.pred)
mean(OJ.train$Purchase != train.pred)

test.pred = predict(svm.poly, OJ.test)
table(OJ.test$Purchase, test.pred)
mean(OJ.test$Purchase != test.pred)

# Summary shows that polynomial kernel produces 452 support vectors, out of which, 232 belong to level CH and remaining 220 belong to level MM. 
# This kernel produces a train error of 17.1% and a test error of 18.1% which are slightly higher than the errors produces by radial kernel but lower than the errors produced by linear kernel.

set.seed(322)
tune.out = tune(svm, Purchase ~ ., data = OJ.train, kernel = "poly", degree = 2, 
                ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
summary(tune.out)

svm.poly = svm(Purchase ~ ., data = OJ.train, kernel = "poly", degree = 2, cost = tune.out$best.parameters$cost)
train.pred = predict(svm.poly, OJ.train)
table(OJ.train$Purchase, train.pred)
mean(OJ.train$Purchase != train.pred)

test.pred = predict(svm.poly, OJ.test)
table(OJ.test$Purchase, test.pred)
mean(OJ.test$Purchase != test.pred)


# Tuning reduces the training error to 15.12% and test error to 17.4% which is worse 
# than radial kernel but slightly better than linear kernel.

# Overall, radial basis kernel seems to be producing minimum misclassification error on both train and test data.


