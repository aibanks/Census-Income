library(tidyverse)
library(caret)
library(ggplot2)
library(rpart)				       

library(rattle)					
library(rpart.plot)			
#library(RColorBrewer)				
#library(party)					
#library(partykit)


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
training_dat %>% group_by(income) %>% 
  summarize(Count=n(), Percent = n()/(length(training_dat$income)))

age_dense <- training_dat %>% ggplot(aes(age, fill = income)) + geom_density(stat ="count") 
age_dense

gender_hist <- training_dat %>% ggplot(aes(sex, fill = income)) + geom_bar(position = position_dodge())
gender_hist

education_hist <- training_dat %>% ggplot(aes(education.num, fill = income)) + geom_histogram(binwidth = 1, position = position_dodge()) #+ theme(axis.text.x = element_text(angle = 90, hjust = 1))
education_hist

race_hist <- training_dat %>% ggplot(aes(race, fill = income)) + geom_histogram(stat="count", position = position_dodge())
race_hist

workclass_hist <- training_dat %>% ggplot(aes(workclass, fill = income)) + geom_histogram(stat = "count", position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
workclass_hist

#hours.per.week_hist <- training_dat %>% ggplot(aes(hours.per.week, fill = income)) +geom_histogram(binwidth = 10, position = position_dodge(), col="black")
#hours.per.week_hist

hours.per.week_freqpoly <- training_dat %>% ggplot(aes(hours.per.week, colour = income)) + geom_freqpoly(binwidth = 6) + scale_x_continuous(limits=c(0, 80))
hours.per.week_freqpoly

marital.status_hist <- training_dat %>% ggplot(aes(marital.status, fill = income)) + geom_histogram(stat = "count", position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
marital.status_hist

native.country_hist <- training_dat %>% ggplot(aes(native.country, fill = income)) + geom_histogram(stat = "count", position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
native.country_hist

native.county_table <- training_dat %>% group_by(native.country) %>%
  summarize(sum(income == "<=50K"), sum(income == ">50K"))
native.county_table

native.county_table <- native.county_table %>% mutate("percent_>50K" = (`sum(income == ">50K")`)/(`sum(income == ">50K")` + `sum(income == "<=50K")`)*100)

native.county_table <- native.county_table %>% arrange(desc(`percent_>50K`))

head(native.county_table, 5)
tail(native.county_table, 5)

########################################
####Data Cleanse########################
########################################
training_dat <- training_dat %>% select(-c(fnlwgt,relationship, education.num))

#########################################
#####Start Models########################
#########################################

#Model 1A: Linear Regression 
training_dat_as.num <- training_dat %>% mutate(income = as.numeric(income == ">50K"))
train_lm <- train(income ~., method = "lm", data = training_dat_as.num)
income_hat_as.num <- predict(train_lm, testing_dat)
income_hat <- ifelse(income_hat_as.num > 0.5, ">50K", "<=50K") %>% factor()

confusionMatrix(data = income_hat, reference = testing_dat$income)
model.results_linear.regression <- confusionMatrix(data = income_hat, reference = testing_dat$income)
#accuracy = 0.8299, sensitivity = 0.9361, specificity = 0.4955


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
#started at 10:14; ended ~10:15

#Model 3B: Regression Tree - adjusting complexity parameter (cp)
train_tree_cp <- train(income ~., 
                       method = "rpart",
                       tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                       data = training_dat)

ggplot(train_tree_cp)
#started at 10:22; ended ~10:24
plot(train_tree_cp$finalModel)
text(train_tree_cp$finalModel, cex = 0.5)

prp(train_tree_cp$finalModel)
fancyRpartPlot(train_tree_cp$finalModel)

income_hat_tree_cp <- predict(train_tree_cp, testing_dat)
confusionMatrix(income_hat_tree_cp, reference = testing_dat$income)
model.results_regression.tree_cp <- confusionMatrix(income_hat_tree_cp, reference = testing_dat$income)

#Accuracy= 0.8545, Sensitivity = 0.9434, Specificity = 0.5745
#Increased all three, but the specificity is still very low.  Will try with forest instead of tree


#Model 4A: Random Forest - Standard (without manually adjusting tuning parameters)
train_random.forest <- train(income ~., 
                             method = "Rborist",
                             data = training_dat)
#started at 10:52; ended 11:17
plot(train_random.forest)

income_hat_random.forest <- predict(train_random.forest, testing_dat)
confusionMatrix(income_hat_random.forest, reference = testing_dat$income)

#Accuracy = 0.759,  Sensitivity = 1.00, Specificity = 0.00

#Model 5A:  Random Forest - Adjusting the minNode
train_random.forest_minNode <- train(income ~ .,
                                     method = "Rborist",
                                     tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)),
                                     data = training_dat)

income_hat_random.forest_minNode <- predict(train_random.forest_minNode, testing_dat)
confusionMatrix(income_hat_random.forest_minNode, reference = testing_dat$income)










