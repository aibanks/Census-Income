#####Update the library sections into "IF REQUIRE"
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(fastAdaboost)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(rattle)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(mboost)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")	

library(tidyverse)
library(caret)
library(ggplot2)
library(rpart)
library(fastAdaboost)
library(rattle)
library(rpart.plot)
library(mboost)
library(plyr)
library(knitr)

install.packages("recipes")


file <- "adult-census-income.zip"

dat <- read.csv(unzip(file))

head(dat)

dim(dat)

sum(is.na(dat$income))

test_index <- createDataPartition(y = dat$income, p = 0.1, list = FALSE)

training_dat <- dat[-test_index,]
testing_dat <- dat[test_index,]

dim(training_dat)
dim(testing_dat)

#####################################
##Data inspection and visualization##
#####################################
training_dat %>% group_by(income) %>% c(Count=n(), Percent = n()/(length(training_dat$income))*100)



age_dense <- training_dat %>% ggplot(aes(age, fill = income)) + geom_density(stat ="count") + ggtitle("Graph 1. Age Distribution")
age_dense

gender_hist <- training_dat %>% ggplot(aes(sex, fill = income)) + geom_bar(position = position_dodge()) + ggtitle("Graph 2. Gender Distribution")
gender_hist

education_hist <- training_dat %>% ggplot(aes(education.num, fill = income)) + geom_histogram(binwidth = 1, position = position_dodge()) + ggtitle("Graph 3. Education Distribution")
education_hist

race_hist <- training_dat %>% ggplot(aes(race, fill = income)) + geom_histogram(stat="count", position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + ggtitle("Graph 4. Race Distribution")
race_hist

race_table <- training_dat %>% group_by(race) %>% 
  dplyr::summarise('Count <=50K' = sum(income == "<=50K"), 'Count >50K' = sum(income == ">50K"))
race_table <- race_table %>% mutate('Percent >50K' = (`Count >50K`) / ((`Count <=50K`) + (`Count >50K`)) *100)
race_table

race.edu_graph <- training_dat %>% ggplot(aes(race, education.num, color = income)) + geom_boxplot()
race.edu_graph

workclass_hist <- training_dat %>% ggplot(aes(workclass, fill = income)) + geom_histogram(stat = "count", position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + ggtitle("Graph 5. Workclass Distribution")
workclass_hist

#hours.per.week_hist <- training_dat %>% ggplot(aes(hours.per.week, fill = income)) +geom_histogram(binwidth = 10, position = position_dodge(), col="black")
#hours.per.week_hist

hours.per.week_freqpoly <- training_dat %>% ggplot(aes(hours.per.week, colour = income)) + geom_freqpoly(binwidth = 6) + scale_x_continuous(limits=c(0, 80)) + ggtitle("Graph 6. Hours Worked Per Week Distribution")
hours.per.week_freqpoly

marital.status_hist <- training_dat %>% ggplot(aes(marital.status, fill = income)) + geom_histogram(stat = "count", position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + ggtitle("Graph 7. Marital Status Distribution")
marital.status_hist

native.country_hist <- training_dat %>% ggplot(aes(native.country, fill = income)) + geom_histogram(stat = "count", position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
native.country_hist

native.county_table <- training_dat %>% group_by(native.country) %>%
  summarize(sum(income == "<=50K"), sum(income == ">50K"))
native.county_table

########################################
####Data Cleanse########################
########################################
training_dat <- training_dat %>% select(-c(fnlwgt,relationship, education))

#########################################
#####Start Models########################
#########################################


#Model 0:  Predicting <=50K for each entry because that's most common in our dataset
training_dat %>% group_by(income) %>% summarize(n=n())


income_pred <- rep("<=50K", length(testing_dat$income)) %>% factor()

confusionMatrix(income_pred, reference = testing_dat$income)

#Accuracy = 0.759, Sensitivity = 1.00, Specificity = 0.00


#Model 1: Linear Regression 
training_dat_as.num <- training_dat %>% mutate(income = as.numeric(income == ">50K"))
train_lm <- train(income ~., method = "lm", data = training_dat_as.num)
income_hat_as.num <- predict(train_lm, testing_dat)
income_hat <- ifelse(income_hat_as.num > 0.5, ">50K", "<=50K") %>% factor()

confusionMatrix(data = income_hat, reference = testing_dat$income)
model.results_linear.regression <- confusionMatrix(data = income_hat, reference = testing_dat$income)
#accuracy = 0.8299, sensitivity = 0.9361, specificity = 0.4955
#increased to:  accuracy = 0.84, sensitivity = 0.943, specificity = 0.5159


#Model 2: Generalized Linear Model (GLM)
train_glm <- train(income ~.,
                   method = "glm",
                   data = training_dat)

income_hat_glm <- predict(train_glm, testing_dat)
#started at 14:08 ended at 14:10

confusionMatrix(income_hat_glm, reference = testing_dat$income)
#Increased to Accuracy = 0.8554, Sensitivity = 0.9373, Specificity = 0.5975


#Model 2: KNN
train_knn <- train(income ~., 
                   method = "knn",
                   data = training_dat)

ggplot(train_knn, highlight = TRUE)

#2.2 KNN with CV and expanded K values
#control <- trainControl(method = "cv", number = 10, p = .9)

train_knn_cv <- train(income ~.,
                      method = "knn",
                      data = training_dat,
                      tuneGrid = data.frame(k = seq(9, 40, 5)))
#trContol = control
ggplot(train_knn_cv, highlight = TRUE)
#taking a very very long time (>30 minutes)
#attempt without cv;  started at 10:01



#Model 3A: Regression Tree - standard (no adjusting tuning parameters)
train_tree <- train(income ~ ., 
                    method = "rpart",
                    data = training_dat)
plot(train_tree, margin = 0.1)
text(train_tree, cex = 0.75)

income_hat_tree <- predict(train_tree, testing_dat)
confusionMatrix(income_hat_tree, reference = testing_dat$income)
#Accuracy = 0.8397, Sensitivity = 0.9543, Specificity = 0.4790

#started at 10:14; ended ~10:15

#Model 3B: Regression Tree - adjusting complexity parameter (cp)
train_tree_cp <- train(income ~., 
                       method = "rpart",
                       tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                       data = training_dat)

ggplot(train_tree_cp, highlight = TRUE)
#started at 10:22; ended ~10:24
plot(train_tree_cp$finalModel)
text(train_tree_cp$finalModel, cex = 0.5)

prp(train_tree_cp$finalModel)
fancyRpartPlot(train_tree_cp$finalModel)

income_hat_tree_cp <- predict(train_tree_cp, testing_dat)
confusionMatrix(income_hat_tree_cp, reference = testing_dat$income)
model.results_regression.tree_cp <- confusionMatrix(income_hat_tree_cp, reference = testing_dat$income)

#Increased to Accuracy = 0.8618, Sensitivity = 0.9620, Specificity = 0.5465


#Model 3C: Regression Tree w. Cross Validation

control <- trainControl(method = "repeatedcv", 
                        repeats = 10)
                      
train_tree_cv <- train(income ~ .,
                       data = training_dat,
                       method = "rpart",
                       tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                       trControl = control)

income_hat_tree_cv <- predict(train_tree_cv, testing_dat)

confusionMatrix(income_hat_tree_cv, testing_dat$income)

#Model 4A: Random Forest - Standard (without manually adjusting tuning parameters)
train_random.forest <- train(income ~., 
                             method = "rf",
                             metric = "Accuracy",
                             trControl = trainControl(),
                             tuneGrid = NULL,
                             data = training_dat)

#started at 10:52; ended 11:17
#started at 15:53;didn't end by 16:54 (manually stopped)
plot(train_random.forest)

income_hat_random.forest <- predict(train_random.forest, testing_dat)
confusionMatrix(income_hat_random.forest, reference = testing_dat$income)

#Accuracy = 0.759,  Sensitivity = 1.00, Specificity = 0.00

#Model 4B:  Random Forest - Adjusting the minNode
train_random.forest_minNode <- train(income ~ .,
                                     method = "Rborist",
                                     tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)),
                                     data = training_dat)

income_hat_random.forest_minNode <- predict(train_random.forest_minNode, testing_dat)
confusionMatrix(income_hat_random.forest_minNode, reference = testing_dat$income)
#started at 11:24; finished by 12:15 (left to get groceries)

#Accuracy = 0.759,  Sensitivity = 1.00, Specificity = 0.00


#Model 4C: Random Forest on smaller training set (30%)

rf_train_index <- createDataPartition(training_dat$income, times = 1, p = 0.05, list = FALSE)
rf_train_dat <- training_dat[rf_train_index,]
rf_else_dat <- training_dat[-rf_train_index,]

train_rf_lim.dat <- train(income ~ .,
                          method = "Rborist",
                          data = rf_train_dat)
#started 17:54; stopped at 18:21


#################################################
###Testing Out Undersampling the Training set####
#################################################

training_dat_undersample <- downSample(training_dat, training_dat$income)

mean(training_dat_undersample$income == "<=50K")

training_dat_undersample <- training_dat_undersample %>% select(-c(Class))





#Model 3A: Regression Tree - standard (no adjusting tuning parameters)
train_tree <- train(income ~ ., 
                    method = "rpart",
                    data = training_dat_undersample)
plot(train_tree, margin = 0.1)
text(train_tree, cex = 0.75)

income_hat_tree <- predict(train_tree, testing_dat)
confusionMatrix(income_hat_tree, reference = testing_dat$income)
#Accuracy = 0.8397, Sensitivity = 0.9543, Specificity = 0.4790
#Accuracy = 0.7157, Sensitivity = 0.6667, Specificity = 0.8701, balanced Acc. = 0.7684



#Model 3B: Regression Tree - adjusting complexity parameter (cp)
train_tree_cp <- train(income ~., 
                       method = "rpart",
                       tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                       data = training_dat_undersample)

#started at 10:22; ended ~10:24

income_hat_tree_cp <- predict(train_tree_cp, testing_dat)
confusionMatrix(income_hat_tree_cp, reference = testing_dat$income)
model.results_regression.tree_cp <- confusionMatrix(income_hat_tree_cp, reference = testing_dat$income)

#Increased to Accuracy = 0.8618, Sensitivity = 0.9620, Specificity = 0.5465
#Acc. = 0.79, Sensitivity = 0.7686, Specificity = 0.8573, Balanced Acc. = 0.8130


##################################################################
###Testing Out Undersampling the Training set and the Test set####
##################################################################

training_dat_undersample <- downSample(training_dat, training_dat$income)

mean(training_dat_undersample$income == "<=50K")

training_dat_undersample <- training_dat_undersample %>% select(-c(Class))


testing_dat_undersample <- downSample(testing_dat, testing_dat$income)

mean(testing_dat_undersample$income == "<=50K")

testing_dat_undersample <- testing_dat_undersample %>% select(-c(Class))

##################################

#Model 3A: Regression Tree - standard (no adjusting tuning parameters)
train_tree <- train(income ~ ., 
                    method = "rpart",
                    data = training_dat_undersample)


income_hat_tree <- predict(train_tree, testing_dat_undersample)
confusionMatrix(income_hat_tree, reference = testing_dat_undersample$income)
#Original:  Accuracy = 0.8397, Sensitivity = 0.9543, Specificity = 0.4790
#Undersampled Training Set:  Accuracy = 0.7157, Sensitivity = 0.6667, Specificity = 0.8701, balanced Acc. = 0.7684
#Undersampled Test and Training Sets:  Accuracy = 0.7707, Sens. = 0.6713, Spec = 0.8701, Balanced Acc. = 0.7707

#Model 5: AdaBoost Classification Trees

train_glmboost <- train(income~.,
                        method = "glmboost",
                        data = training_dat)

income_hat_glmboost <- predict(train_glmboost, testing_dat)
confusionMatrix(income_hat_glmboost, reference = testing_dat$income)

#started at 18:43  






















