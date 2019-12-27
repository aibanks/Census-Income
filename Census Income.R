#####Install the required packages (if not already installed), and load them in.
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


##Read in the file
file <- "adult-census-income.zip"
dat <- read.csv(unzip(file))

##Check the first entries and the dimensions of the dataset
head(dat)
dim(dat)

##Check for any missing values or NA values in the dataset
sum(is.na(dat$income))

##Set the seed for reproducibility, then partition the original dataset into a training dataset
## with 90% of the original dataset and a testing dataset with 10% of the original dataset.
set.seed(1993, sample.kind = "Rounding")
test_index <- createDataPartition(y = dat$income, p = 0.1, list = FALSE)
training_dat <- dat[-test_index,]
testing_dat <- dat[test_index,]

##Check the dimensions of the training and testing datasets
dim(training_dat)
dim(testing_dat)

#####################################
##Data inspection and visualization##
#####################################

#Table showing the distribution of income entries among the training dataset
training_dat %>% group_by(income) %>% c(Count=n(), Percent = n()/(length(training_dat$income))*100)

#Graph showing the age distribution of the income categories
age_dense <- training_dat %>% ggplot(aes(age, fill = income)) + geom_density(stat ="count") + ggtitle("Graph 1. Age Distribution")
age_dense

#Histogram of gender of the income categories
gender_hist <- training_dat %>% ggplot(aes(sex, fill = income)) + geom_bar(position = position_dodge()) + ggtitle("Graph 2. Gender Distribution")
gender_hist

#Histogram of the years of education of the income categories
education_hist <- training_dat %>% ggplot(aes(education.num, fill = income)) + geom_histogram(binwidth = 1, position = position_dodge()) + ggtitle("Graph 3. Education Distribution")
education_hist

#Histogram of the races of the income categories
race_hist <- training_dat %>% ggplot(aes(race, fill = income)) + geom_histogram(stat="count", position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + ggtitle("Graph 4. Race Distribution")
race_hist

#Table of the races of the income categories
race_table <- training_dat %>% group_by(race) %>% 
  dplyr::summarise('Count <=50K' = sum(income == "<=50K"), 'Count >50K' = sum(income == ">50K"))
race_table <- race_table %>% mutate('Percent >50K' = (`Count >50K`) / ((`Count <=50K`) + (`Count >50K`)) *100)
race_table

#Histogram of the workclasses of the income categories
workclass_hist <- training_dat %>% ggplot(aes(workclass, fill = income)) + geom_histogram(stat = "count", position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + ggtitle("Graph 5. Workclass Distribution")
workclass_hist

#Graph of the number of hours worked per week of the income categories
hours.per.week_freqpoly <- training_dat %>% ggplot(aes(hours.per.week, colour = income)) + geom_freqpoly(binwidth = 6) + scale_x_continuous(limits=c(0, 80)) + ggtitle("Graph 6. Hours Worked Per Week Distribution")
hours.per.week_freqpoly

#Histogram of the marital status of the income categories
marital.status_hist <- training_dat %>% ggplot(aes(marital.status, fill = income)) + geom_histogram(stat = "count", position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + ggtitle("Graph 7. Marital Status Distribution")
marital.status_hist

########################################
####Data Cleanse########################
########################################
#Removing the columns that I do not want to use in the model
training_dat <- training_dat %>% select(-c(fnlwgt,relationship, education))

#########################################
#####Start Models########################
#########################################

#Model 1:  Predicting <=50K for each entry because that's most common in our dataset
training_dat %>% group_by(income) %>% summarize(n=n())
income_pred <- rep("<=50K", length(testing_dat$income)) %>% factor()
confusionMatrix(income_pred, reference = testing_dat$income)

results1 <- confusionMatrix(income_pred, reference = testing_dat$income)
'Model 1' <- as.table(c(results1$byClass["F1"] , results1$overall["Accuracy"], 
                                results1$byClass["Sensitivity"] , results1$byClass["Specificity"]))
results_table <- rbind(`Model 1`)
results_table

#Model 2: Linear Regression 
training_dat_as.num <- training_dat %>% mutate(income = as.numeric(income == ">50K"))
train_lm <- train(income ~., method = "lm", data = training_dat_as.num)
income_hat_as.num <- predict(train_lm, testing_dat)
income_hat <- ifelse(income_hat_as.num > 0.5, ">50K", "<=50K") %>% factor()

confusionMatrix(data = income_hat, reference = testing_dat$income)
model.results_linear.regression <- confusionMatrix(data = income_hat, reference = testing_dat$income)

results2 <- confusionMatrix(data = income_hat, reference = testing_dat$income)
'Model 2' <- as.table(c(results2$byClass["F1"] , results2$overall["Accuracy"], 
                                results2$byClass["Sensitivity"] , results2$byClass["Specificity"]))
results_table <- rbind(results_table, `Model 2`)
results_table


#Model 3: Generalized Linear Model (GLM)
train_glm <- train(income ~.,
                   method = "glm",
                   data = training_dat)
income_hat_glm <- predict(train_glm, testing_dat)
confusionMatrix(income_hat_glm, reference = testing_dat$income)

results3 <- confusionMatrix(income_hat_glm, reference = testing_dat$income)
'Model 3' <- as.table(c(results3$byClass["F1"] , results3$overall["Accuracy"], 
                                results3$byClass["Sensitivity"] , results3$byClass["Specificity"]))
results_table <- rbind(results_table, `Model 3`)
results_table

#Model 4A: Regression Tree - standard (no adjusting tuning parameters)
train_tree <- train(income ~ ., 
                    method = "rpart",
                    data = training_dat)
plot(train_tree, margin = 0.1)
text(train_tree, cex = 0.75)

income_hat_tree <- predict(train_tree, testing_dat)
confusionMatrix(income_hat_tree, reference = testing_dat$income)

results4a <- confusionMatrix(income_hat_tree, reference = testing_dat$income)


'Model 4A' <- as.table(c(results4a$byClass["F1"] , results4a$overall["Accuracy"], 
                               results4a$byClass["Sensitivity"] , results4a$byClass["Specificity"]))

results_table <- rbind(results_table, `Model 4A`)
results_table

#Model 4B: Regression Tree - adjusting complexity parameter (cp)
train_tree_cp <- train(income ~., 
                       method = "rpart",
                       tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                       data = training_dat)

ggplot(train_tree_cp, highlight = TRUE) + ggtitle("Graph 8. Estimated Accuracy vs Complexity Parameter in Model 4B")
prp(train_tree_cp$finalModel, main = "Figure 1. Decision Tree of Model 4B")

income_hat_tree_cp <- predict(train_tree_cp, testing_dat)
confusionMatrix(income_hat_tree_cp, reference = testing_dat$income)
results4b <- confusionMatrix(income_hat_tree_cp, reference = testing_dat$income)


'Model 4B' <- as.table(c(results4b$byClass["F1"] , results4b$overall["Accuracy"], 
                               results4b$byClass["Sensitivity"] , results4b$byClass["Specificity"]))

results_table <- rbind(results_table, `Model 4B`)
results_table
