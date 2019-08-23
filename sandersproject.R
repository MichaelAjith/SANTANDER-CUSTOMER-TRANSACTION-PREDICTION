#remove all the objects stored
rm(list=ls())

#set current working directory
setwd("D:/MBA REF/edwisor/project/sanders")

#Current working directory
getwd()

# importing all required library
Packages <- c("ggplot2","ggpubr","randomForest","caret", "class", "e1071", 
              "rpart", "DMwR","usdm","dplyr","caTools",
              "C50","RODBC","outliers","car","Boruta","Metrics","ggthemes",
              "DataCombine","inTrees","pROC","xgboost")

lapply(Packages, library, character.only = TRUE)

# Reading/Loading the csv file
cust<-read.csv("train.csv", header =  TRUE , stringsAsFactors = FALSE)


# *****************************  Exploratory Data Analysis ****************************        


# ******************** Understanding the data  ********************************** 

# checking datatypes of all columns
class(cust)
dim(cust)
head(cust)
names(cust)
str(cust)
summary(cust)

#checking the target variable of the dataset
hist(cust$target , col = "red")


#removing one variable  "ID_Code" the target variable from the training set
cust <- cust[, -1, drop = FALSE]

# ********************** Outlier Analysis *************************

#We are using the cooks Distance
#first we will build the model and on that bases we will build the model 

mod <- lm(target ~., data = cust)
csd<-cooks.distance(mod)

plot(csd, pch = 1, cex = 2, main = "influential points")
abline(h = 4*mean(csd, na.rm = T), col = "red")
text(x=1:length(csd)+1, y=csd, labels = ifelse(csd >4*mean(csd, na.rm = T), names(csd), ""), col = "blue")
influential <- as.numeric(names(csd)[(csd >4*mean(csd, na.rm = T))])
head(cust[influential,])

#we doing outliers test using car package 
car:: outlierTest(mod)
#it show that 94184 in this row it has the most extreme values  

#imputation of the outliers we are using the capping function
x<- as.data.frame(cust)
caps <- data.frame(apply(cust,2, function(x){
  quantiles <- quantile(x, c(0.25, 0.75))
  x[x < quantiles[1]] <- quantiles[1]
  x[x > quantiles [2]] <- quantiles[2]
}))

caps

# ***********************  Missing value analysis *********************** 

#checking the count of all the na's in data
sum(is.na(cust))

# **************************  Feature Selection  ****************************      

#feature selection of the data using the Boruta method
#Boruta method 
set.seed(123)
btrain <- Boruta(target~., data = cust, doTrace = 2)
print(btrain)


#Boruta performed 99 iterations in 7.526 hrs.
#2 attributes confirmed important: var_170, var_99;
#196 attributes confirmed unimportant: var_0, var_1, var_10, var_100, var_101 and 191 more;
#2 tentative attributes left: var_114, var_92;

#Ploting graph 
plot(btrain, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(btrain$ImpHistory),function(i)
  btrain$ImpHistory[is.finite(btrain$ImpHistory[,i]),i])
names(lz) <- colnames(btrain$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(btrain$ImpHistory), cex.axis = 0.7)

#tentative variable test 
final.boruta <- TentativeRoughFix(btrain)
print(final.boruta)
getSelectedAttributes(final.boruta, withTentative = F)
boruta.df <- attStats(final.boruta)

# **************************  Feature Scaling  ****************************      
#Feature scaling only for  Random forest
#selecting variables 
cust1 <- data.frame(cust$target, cust$var_99, cust$var_170)
colnames(cust1)
View(cust1)

#normalization method
cnames = c("cust.var_99", "cust.var_170")
for (i in cnames) {
  cust1 [,i] = (cust1[,i] - min(cust1[,i]))/ (max(cust1[,i] - min(cust1[,i])))
  
}
head(cust1)

# *****************************  Building models  ********************************                
rm(list= ls()[!(ls() %in% c('cust','cust1'))])
set.seed(321)

intrain <- createDataPartition(y = cust$target, p = 0.7, list = FALSE)
# ?createDataPartition part of caret package
training<- cust[intrain,]
hist(training$target)
length(training$target)
length(which(training$target==0))
length(which(training$target==1))

testing <- cust[-intrain,]
hist(testing$target)
length(testing$target)
length(which(testing$target==0))
length(which(testing$target==1))
#View(testing)
# model preparation. Exclude ID column from dataset
trainingExID <- training[]
head(trainingExID)
View(trainingExID)

# ***************************  Logistic Regression  ****************************

# model preparation using Logistic regression
model1 <- glm(trainingExID$target~., data = trainingExID, family = binomial(link = 'logit'))
summary(model1)
model1
# prediction on test data
pred1 <- predict(model1, testing[,-1])
head(pred1) # gives prediction probabilities
pred1 <- as.numeric(pred1 > 0.5)
summary(pred1)
hist(pred1)


#error matrix 

conmatrix <- table(testing$target, pred1)
confusionMatrix(conmatrix)

#Recall,Precision and Accuracy
recall<-53668/(53668+4919)
recall
precision<- 53668/(53668+321)
precision
accuracy<- (53668+1092)/60000
accuracy  
#Recall for logistic regression is 91.6%
#Precision  for logistic regression is 99.4%
#Accuracy for logistic regression is 91.2%  
# model performance
plot.roc(testing$target, pred1)
auc(roc(testing$target, pred1)) # provides AUC value which is ~0.5879

# ****************************** Decision Tree  ********************

# model preparation using Decision Tree

model2 <-C5.0(as.factor(trainingExID$target)~., data = trainingExID ,trails=100,rules = TRUE)
write(capture.output(summary(model2)), "summary_cust.text")
pred2 <- predict(model2, testing[,-1])

#error matrix 

conmatrix <- table(testing$target, pred2)
confusionMatrix(conmatrix)

#Recall,Precision and Accuracy
recall<-53386/(53386+5624)
recall

precision<- 53386/(53386+603)
precision
accuracy<- (53386+387)/60000
accuracy 
#Recall for Decision Tree is 90.4%
#Precision  for Decision Tree is 98.8%
#Accuracy for Decision Tree is 89.62%  

# model performance
pred2<-as.numeric(pred2)
plot.roc(testing$target, pred2)
auc(roc(testing$target, pred2)) # provides AUC value which is ~0.5266

# **********************************  Random Forest  ************************* 
# model preparation using Random Forest  
cust1<- as.data.frame(cust1)

train_index = sample(1:nrow(cust1), 0.8*nrow(cust1))
train_data = cust1[train_index,]
test_data = cust1[-train_index,]

model3 <- randomForest(as.factor(cust.target)~., train_data, importance = TRUE)
treelist <- RF2List(model3)
rules <- extractRules(treelist, train_data[,-1])
rules[5,]

readrule <- presentRules(rules, colnames(train_data))
head(readrule)

rulematrix <- getRuleMetric(rules, train_data[,-1], train_data$cust.target)
head(rulematrix)

#error matrix
pred3 <- predict(model3, test_data[,-1])

confmt <- table(test_data$cust.target, pred3)

confusionMatrix(confmt)

summary(model3)

#Recall,Precision and Accuracy
recall<-35683/(35683+3998)
recall

precision<- 35683/(35683+275)
precision
accuracy<- (35683+44)/40000
accuracy  
#Recall for Random Forest is 89.9%
#Precision  for Random Forest is 99.2%
#Accuracy for Random Forest is 89.3%  

# model performance
pred3 <- as.numeric(pred3)
plot.roc(test_data$cust.target, pred3)
auc(roc(test_data$cust.target, pred3))# provides AUC value which is ~0.5015


# ***********************************  XGBoost ********************************      


# model preparation using XGBoost

model4 <- xgboost(data = as.matrix(training[,-1:-2]), label = as.matrix(training$target), max_depth = 2, eta = 1, nthread = 6, nrounds = 3000, maximize = T, print_every_n = 100, objective = "binary:logistic")
summary(model4)
model4

# prediction on test data
pred4 <- predict(model4,as.matrix(testing[,-1:-2]))
head(pred4) # gives prediction probabilities
pred4 <- as.numeric(pred4 > 0.5)
summary(pred4)
hist(pred4)
length(which(pred4 == 1))
length(which(pred4 == 0))
hist(pred4)

#error matrix 

conmatrix <- table(testing$target, pred4)
confusionMatrix(conmatrix)
#Recall,Precision and Accuracy
recall<-51701/(51701+3610)
recall
precision<- 51701/(51701+2288)
precision
accuracy<- (51701+2401)/60000
accuracy  
#Recall for  XGBoost is 93.4%
#Precision  for XGBoost is 95.7%
#Accuracy for  XGBoost is 90.9% 

# model performance

plot.roc(testing$target, pred4)
auc(roc(testing$target, pred4)) # provides AUC value which is ~0.6785

#************************ Hyperparameter tuning **************************************             

#***************************** Tuning XGBoost  ****************************

# preparing XGB matrix
dtrain <- xgb.DMatrix(data = as.matrix(training[,-1:-2]), label = as.matrix(training$target))
# parameters
params <- list(booster = "gbtree",
               objective = "binary:logistic",
               eta=0.02,
               #gamma=80,
               max_depth=2,
               min_child_weight=1, 
               subsample=0.5,
               colsample_bytree=0.1,
               scale_pos_weight = round(sum(!training$target) / sum(training$target), 2))
set.seed(123)
xgbcv <- xgb.cv(params = params, 
                data = dtrain, 
                nrounds = 30000, 
                nfold = 5,
                showsd = F, 
                stratified = T, 
                print_every_n = 100, 
                early_stopping_rounds = 500, 
                maximize = T,
                metrics = "auc")

 cat(paste("Best iteration:", xgbcv$best_iteration))

set.seed(123)
xgb_model <- xgb.train(
  params = params, 
  data = dtrain, 
  nrounds = xgbcv$best_iteration, 
  print_every_n = 100, 
  maximize = T,
  eval_metric = "auc")

#view variable importance plot
imp_mat <- xgb.importance(feature_names = colnames(training[,-1:-2]), model = xgb_model)
xgb.plot.importance(importance_matrix = imp_mat[1:30])



# prediction on test data
pred5 <- predict(xgb_model, as.matrix(testing[,-1:-2]))
head(pred5) # gives prediction probabilities
pred5 <- as.numeric(pred5 > 0.5)
summary(pred4)
length(which(pred5 == 1))
length(which(pred5 == 0))
hist(pred5)

#error matrix 

conmatrix <- table(testing$target, pred5)
confusionMatrix(conmatrix)


#Recall,Precision and Accuracy
recall<-45847/(45847+1355)
recall

precision<- 45847/(45847+8142)
precision
accuracy<- (45847+4656)/60000
accuracy
#Recall for Tunned XGBoost is 97.1%
#Precision  for Tunned XGBoost is 84.9%
#Accuracy for Tunned XGBoost is 84.17% 

# model performance
plot.roc(testing$target, pred5)
auc(roc(testing$target, pred5)) # provides AUC value which is ~0.8119