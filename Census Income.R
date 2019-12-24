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

#accuracy = 0.8299, sensitivity = 0.9361, specificity = 0.4955
#increased to:  accuracy = 0.84, sensitivity = 0.943, specificity = 0.5159


#Model 3: Generalized Linear Model (GLM)
train_glm <- train(income ~.,
                   method = "glm",
                   data = training_dat)

income_hat_glm <- predict(train_glm, testing_dat)
#started at 14:08 ended at 14:10

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

ggplot(train_tree_cp, highlight = TRUE)
#started at 10:22; ended ~10:24
plot(train_tree_cp$finalModel)
text(train_tree_cp$finalModel, cex = 0.5)

prp(train_tree_cp$finalModel)
fancyRpartPlot(train_tree_cp$finalModel)

income_hat_tree_cp <- predict(train_tree_cp, testing_dat)
confusionMatrix(income_hat_tree_cp, reference = testing_dat$income)
results4b <- confusionMatrix(income_hat_tree_cp, reference = testing_dat$income)


'Model 4B' <- as.table(c(results4b$byClass["F1"] , results4b$overall["Accuracy"], 
                               results4b$byClass["Sensitivity"] , results4b$byClass["Specificity"]))

results_table <- rbind(results_table, `Model 4B`)
results_table



