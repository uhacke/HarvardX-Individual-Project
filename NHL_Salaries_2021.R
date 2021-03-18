library(caret)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(corrplot) 
library(randomForest)
library(extraTrees)
library(rJava)
library(gbm)
library(xgboost)
library(h2o)



setwd("~/Datascience/Data Science Courses/HarvardX Course/Individual project")

# Data for the 2020/21 season from HockeyReference (https://www.hockey-reference.com/friv/current_nhl_salaries.cgi) and NHL (http://www.nhl.com/stats/skaters).

dat = read.csv("NHL_players_stats_merged.csv", header = T)

head(dat)

# Legend: S.C=Skater shoots, Pos=Player position, GP=Games played, G=Goals, A=Assists, P=Points, plus_minus, PIM=Penalty minutes, PperGP=Points per game played, EVG=Even strength goals, EVP=Even strength points, 
# PPG=Powerplay goals, PPP=Powerplay points, SHG=Shorthanded goals, SHP=Shorthanded points, OTG=Overtime goals, GWG=Game-winning goals, S=Shots, TOI=Time on ice 

# Wrangling and getting descriptive numbers
is.na(dat$salary)

cat("\n\n Sort data frame by salary in descending order\n")
# sort data frame by salary in descending order
dat_sorted <- dat[with(dat, order(-dat$Salary)), ]

print(dat_sorted)

# Make sure numbers are in numeric format
dat_sorted$Salary <- as.numeric(dat_sorted$Salary)
dat_sorted$GP <- as.numeric(dat_sorted$GP)
dat_sorted$G <- as.numeric(dat_sorted$G)
dat_sorted$A <- as.numeric(dat_sorted$A)
dat_sorted$plus_minus <- as.numeric(dat_sorted$plus_minus)
dat_sorted$PIM <- as.numeric(dat_sorted$PIM)
dat_sorted$PperGP<- as.numeric(dat_sorted$PperGP)
dat_sorted$EVG <- as.numeric(dat_sorted$EVG)
dat_sorted$EVP <- as.numeric(dat_sorted$EVP)
dat_sorted$PPG <- as.numeric(dat_sorted$PPG)
dat_sorted$PPP <- as.numeric(dat_sorted$PPP)
dat_sorted$SHG<- as.numeric(dat_sorted$SHG)
dat_sorted$SHP <- as.numeric(dat_sorted$SHP)
dat_sorted$OTG <- as.numeric(dat_sorted$OTG)
dat_sorted$GWG <- as.numeric(dat_sorted$GWG)
dat_sorted$S<- as.numeric(dat_sorted$S)
dat_sorted$TOI<- as.numeric(dat_sorted$TOI)

# Examine the structure of the dat dataset
str(dat)

# Create a summary for the dat dataset
summary(dat)

# Separate forwards and defenders; first forwards
Forwards<-subset(dat, dat$Pos !='D')
Forwards

# Now create a dataframe with the defensemen
Def<-subset(dat, dat$Pos == 'D')
Def

# Compare the salary in the Forwards and Defensemen datasets with a boxplot
p1 = dat$Salary[which(dat$Pos !='D')]/1000000
p2 = dat$Salary[which(dat$Pos =='D')]/1000000
par(mfrow=c(1,2))
boxplot(p1)
boxplot(p2)
par(mfrow=c(1,1))
boxplot(p1,p2, main = "Forwards vs defensemen", ylab = "Salary in USD (millions)", names = c("Forwards", "Defensemen"), col = "#69b3a2")

# Make salary histogram
Histo <- ggplot(dat, aes(x=Salary)) + 
  geom_histogram(fill= "#69b3a2", col = "black")
Histo

# Compare the salary in the Forwards and Defensemen datasets with numbers
summary(Forwards)
mean(Forwards$Salary)
median(Forwards$Salary)
max(Forwards$Salary)
min(Forwards$Salary)

summary(Def)
mean(Def$Salary)
median(Def$Salary)
max(Def$Salary)
min(Def$Salary)

# Build correlation matrix of all parameters. First exclude non-numeric data.
# For forwards
sapply(Forwards, is.numeric)
Forwards_num_data <- Forwards[, sapply(Forwards, is.numeric)]
cor(Forwards_num_data, use = "complete.obs", method = "pearson")
corrplot(cor(Forwards_num_data), method = "circle", type = "upper", title = "Forwards") 

# For defensemen
sapply(Def, is.numeric)
Def_num_data <- Def[, sapply(Def, is.numeric)]
cor(Def_num_data, use = "complete.obs", method = "pearson")
corrplot(cor(Def_num_data), method = "circle", type = "upper", title = "Defensemen")

# Focus on Forwards and remove predictors that show no strong correlation and name this dataframe FWD
FWD <- Forwards_num_data %>% select(Salary, GP, G, P, PperGP, PPG, PPP, S, TOI)
head(FWD)
pairs(FWD)

# Create a linear model with promising predictors
FWDModel = lm(Salary~ GP+P+PPP+S+TOI, data=FWD)  
summary(FWDModel)         

# Plot residuals - Salaries are not normally distributed
plot(FWDModel$residuals)
abline(h=0, lty=2)

# I tried log-transforming salaries and it doesn't work either. Hence, I will abandon linear regression and will instead focus on other techniques.

options(java.parameters = "-Xmx4g")
# I will use RMSE as my metric, and will define it as follows:
rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

# Then we split the dataset, 80:20 in this case, i.e., 80% of the data will go to the training set and 20% will go to the test set.
FWD1 = sort(sample(nrow(FWD), nrow(FWD)*.8))

#creating training data set by selecting the output row values
train<-FWD[FWD1,]

#creating test data set by not selecting the output row values
test<-FWD[-FWD1,] 

head(train)
nrow(train)
head(test)
nrow(test)

# Random Forest
# We set up cross-validation 5 fold and create a grid of mtry values (Here, trying all possible values.)
cv_5 = trainControl(method = "cv", number = 5)
rf_grid =  expand.grid(mtry = 1:8)

rf_fit = train(Salary ~ ., data = train,
               method = "rf",
               trControl = cv_5,
               tuneGrid = rf_grid)
rf_fit$bestTune
plot(rf_fit)
rmse(predict(rf_fit, test), test$Salary)

# The resulting test RMSE with mtry = 2 is 1791422

# Extremely Randomized Trees - not running on this computer because Java version is incompatible, ran on other PC previously.
# et_grid =  expand.grid(mtry = 1:8, numRandomCuts = 1:10)
# et_fit = train(Salary ~ ., data = train,
#              method = "extraTrees",
#              trControl = cv_5,
#              tuneGrid = et_grid,
#              numThreads = 4)

# et_fit$bestTune
# plot(et_fit)
# rmse(predict(et_fit, test), test$Salary)

# The resulting test RMSE with mtry = 6 is 2014886

# Generalized Boosted Regression Modeling, gbm 
gbm_fit = train(Salary ~ ., data = train,
                method = "gbm",
                trControl = cv_5,
                verbose = FALSE,
                tuneLength = 10)

gbm_fit$bestTune
plot(gbm_fit)
rmse(predict(gbm_fit, test), test$Salary)

# The test RMSE is 1932314

# Extreme Gradient Boosting, xgboost. I get tons of warning messages.
xgb_fit = train(Salary ~ ., data = train,
                method = "xgbTree",
                trControl = cv_5,
                verbose = FALSE,
                tuneLength = 10,
                numThreads = 8)

xgb_fit$bestTune
# nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
# 51      50         1 0.3     0              0.6                1 0.7777778

plot(xgb_fit)
rmse(predict(xgb_fit, test), test$Salary)
# # The test RMSE is 1773959

# Finally, let's see how h2o models compare
h2o.init(nthreads = -1)

Fwd_concise = read.csv("Fwd_concise.csv", header = T)
head(Fwd_concise)

names(Fwd_concise) <- c("Player", "Salary", "GP", "G", "P", "PperGP", "PPG", "PPP", "S", "TOI")
head(names)

FWD_h2o <- h2o.importFile("FWD_concise.csv")

# Create the training dataset and test dataset (80% and 20%)
partitions <- h2o.splitFrame(data = as.h2o(FWD_h2o),
                             ratios = c(0.8),
                             seed = 1)

data_train_h2o   <- h2o.assign(data = partitions[[1]], key = "data_train_H2O")
data_test_h2o    <- h2o.assign(data = partitions[[2]], key = "data_test_H2O")
y1  <-  "Salary" 
x1  <-  setdiff(names(data_train_h2o), y1)

# Applies the H2O AutoML Machine Learning Platform
aml <- h2o.automl(x = x1, y = y1,
                  training_frame = data_train_h2o,
                  validation_frame = data_test_h2o,
                  stopping_metric = "RMSE",
                  seed = 1,
                  sort_metric = "RMSE")

lb <- aml@leaderboard
print(lb, n = nrow(lb))
aml@leader

# test prediction of the leader model
pred <- h2o.predict(aml, data_test_h2o)

# retrieve the leaderboard
lb <- h2o.get_leaderboard(object = aml, extra_columns = 'ALL')
lb

# The best model was: XRT_1_AutoML_20210318_190011 with an rmse of 1923438
                                   