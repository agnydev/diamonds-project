##########################################################
# Report: Diamonds Price Prediction by Its Characteristics
# Date: 25/10/2021
# Author: Inga Aritenco 
##########################################################

############################################################################################ 
# Goal: 
# Prepare Diamonds dataset.
# Explore and visualize data.
# Discover how each diamonds attribute affects on price.
# Predict diamonds price by its characteristics.
# Use two non-trivial machine learning algorithms: 
# 1. Decision Tree
# 2. Random Forest.
# Split diamonds dataset into two subsets: one for training, and one for testing.
# Train set should be divided by 80%, and the test set should be split by 20% respectively.
# Show a confusion matrix in the end of predictions for each subset.
# Check the accuracy for each algorithm.

## Build neural network with `neuralnet` package, as an addendum* to the report. 
## Before creating the neural network normalize data using `min-max normalization`.
## Split data into train(80%) and test(20%) sets.
## Validate the neural model.
## Compute the accuracy of the validation set.
############################################################################################      

##### ANALYSIS #####

##### Data Preparation #####

# Download all required packages:
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(neuralnet)) install.packages("neuralnet", repos = "http://cran.us.r-project.org")
if(!require(lattice)) install.packages("lattice", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
if(!require(varhandle)) install.packages("varhandle", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

# If all the packages are pre-installed, use them
library(tidyverse)
library(dplyr)
library(caret)
library(randomForest)
library(neuralnet)
library(lattice)
library(ggplot2)
library(GGally)
library(RColorBrewer)
library(varhandle) # for converting categorical data into binary 
library(gridExtra)

# Download CSV file from github account:
# https://raw.githubusercontent.com/oryme/diamonds-project/main/diamonds.csv
# The dataset was originally taken from Kaggle Platform for Data Scientists
# https://www.kaggle.com/shivam2503/diamonds

diamonds <- read.csv(file = 
                       "https://raw.githubusercontent.com/oryme/diamonds-project/main/diamonds.csv")

# For commodity purposes in future data analysis convert categorical values 
# `cut`, `color`, `clarity` to `as.factor`
diamonds <- mutate(diamonds,
                   cut = as.factor(cut),
                   color = as.factor(color),
                   clarity = as.factor(clarity))


##### Data Exploration #####

# Data dimensions
dim(diamonds)

# First look at `diamonds` dataset 
str(diamonds)

# First five rows of the dataset
head(diamonds)

# Make sure if there are any missing values in the dataframe 
sum(is.na(diamonds))

# Count the number of occurrences for each diamond attribute 
table(diamonds$cut)
table(diamonds$clarity)
table(diamonds$color)

### DATA PREPROCESSING ###

# Exclude unnecessary column `X` 
diamonds <- subset(diamonds, select = -c(X))
# Check how the data looks like after excluding `X` column
head(diamonds)
# Check the summary statistics of the diamonds dataset 
summary(diamonds)

## Get the information about diamonds carat 

# Minimum value of the diamonds carat
min(diamonds$carat)
# Average value of the diamonds carat
mean(diamonds$carat)
# Maximum value of the diamonds carat
max(diamonds$carat)

## Explore diamonds price by it's features

# Amount of diamonds that cost less than $500 
summary(diamonds$price < 500)

# Amount of diamonds that cost less than $250
summary(diamonds$price < 250)

# Amount of diamonds that cost $15,000 or more 
summary(diamonds$price >= 15000)

# The most expensive diamond in the dataset
subset(diamonds, price == max(price))

# A diamond's price average 
mean(diamonds$price)

# The most low-cost diamond
subset(diamonds, price == min(price))


##### DATA VISUALIZATION #####

## Categorical features visualization 

# Create bar plots to count the categorical features of the diamonds
ggplot(diamonds) + 
  geom_bar(aes(cut, fill = cut)) + 
  labs(title = "Diamonds prevalence by cut") + 
  scale_fill_brewer(palette = "Reds")

ggplot(diamonds) +
  geom_bar(aes(clarity, fill = clarity)) + 
  labs(title = "Diamonds prevalence by clarity") + 
  scale_fill_brewer(palette = "Blues")

ggplot(diamonds) + 
  geom_bar(aes(color, fill = color)) +
  labs(title = "Diamonds prevalence by color") +
  scale_fill_brewer(palette = "Greens")


# Summary statistics for each attribute of the diamonds cut by price
by(diamonds$price, diamonds$cut, summary)
# Price density plot by cut
ggplot(diamonds) + 
  geom_density(aes(x = price, fill = cut)) + scale_x_continuous(limit = c(300, 18900),
                                                                   breaks = seq(300, 18900, 2100)) + 
  facet_wrap(~cut, ncol = 1) + 
  theme_bw() + 
  ggtitle("Price density plot by cut")


# Summary statistics for each attribute of the diamonds clarity by price
by(diamonds$price, diamonds$clarity, summary)
# Price density plot by clarity
ggplot(diamonds) + 
  geom_density(aes(x = price, fill = clarity)) + scale_x_continuous(limit = c(300, 18900),
                                                                breaks = seq(300, 18900, 2100)) + 
  facet_wrap(~clarity, ncol = 1) + 
  theme_bw() + 
  ggtitle("Price density plot by clarity")


# Summary statistics for each attribute of the diamonds color by price
by(diamonds$price, diamonds$color, summary)
# Price density plot by color
ggplot(diamonds) + 
  geom_density(aes(x = price, fill = color)) + scale_x_continuous(limit = c(300, 18900),
                                                                    breaks = seq(300, 18900, 2100)) + 
  facet_wrap(~color, ncol = 1) + 
  theme_bw() + 
  ggtitle("Price density plot by color")


## Continuous values visualization

# Define the distribution between numeric variables using histogram and density
# Explore the distribution between `x`, `y`, `z`
# Plot histogram distributions between those variables

# Histogram and density for `x` distribution
ggplot(diamonds) + 
  geom_histogram(mapping = aes(x = x, y = ..density..), binwidth = 0.01) +
  geom_density(aes(x = x, y = ..density..),
               lwd = 0.5, color = 5, fill = 5, alpha = 0.25) +
  ggtitle("Histogram and Density plot
        for x distribution")

# Histogram and density for `y` distribution
ggplot(diamonds) + 
  geom_histogram(mapping = aes(x = y, y = ..density..), binwidth = 0.01) +
  geom_density(aes(x = y, y = ..density..),
               lwd = 0.5, color = 5, fill = 5, alpha = 0.25) +
  ggtitle("Histogram and Density plot
        for y distribution")

# Histogram and density for `z` distribution
ggplot(diamonds) + 
  geom_histogram(mapping = aes(x = z, y = ..density..), binwidth = 0.01) +
  geom_density(aes(x = z, y = ..density..),
               lwd = 0.5, color = 5, fill = 5, alpha = 0.25) +
  ggtitle("Histogram and Density plot
        for z distribution")

# Some of  `x`, `y`, `z` have zero values 
# It was observed in summary statistics of the entire diamonds dataset
# There are also the outliers 
# Let's verify the above written observations:
# Filter each value and equalize them to zero
filter(diamonds, x == 0 | y == 0 | z == 0) 
# Drop null observations
diamonds <- subset(diamonds, x!=0 & y!=0 & z!=0)




## Show correlation matrix to see interconnection between continuous variables 
ggcorr(diamonds, 
       label = TRUE,
       label_alpha = FALSE)

##### DATA PREPARATION #####

# Cutting the diamond price and carats into ordinal factors
diamonds$fprice <- as.numeric(cut(diamonds$price, 
                                  seq(from = 0, to = 50000, by = 4000)))

# Convert factored price into ordinal factor
diamonds$fprice <- ordered(diamonds$fprice)

# Diamond's carat is a cut by range 0.5 
# `fcarat` is a factored carat
diamonds$fcarat <- as.numeric(cut(diamonds$carat, 
                                  seq(from = 0, to = 6, by = 0.1)))

# Convert factored carat into ordinal factor
diamonds$fcarat <- ordered(diamonds$fcarat)


# Convert categorical variables `cut`, `clarity`, `color` to binary
binary_cut <- to.dummy(diamonds$cut, "cut")

binary_clarity <- to.dummy(diamonds$clarity, "clarity")

binary_color <- to.dummy(diamonds$color, "color")

# Check converted data
head(binary_cut)
head(binary_clarity)
head(binary_color)

# Create new `diamonds2` dataframe
# Add newly created columns
diamonds2 <- cbind(diamonds, binary_cut, binary_clarity, binary_color)

# Exclude from the new dataset `cut`,`color`,`clarity`, `depth`, `table` and
# `x`, `y`, `z` columns
diamonds2 <- subset(diamonds2, select = -c(cut, color, clarity, depth, table,
                                           x, y, z))

# Check new structure of the new `diamonds2` dataset
head(diamonds2)



# Assign the number of observations to `10000` to make the dataset smaller 
num_obs <- 10000

# Sample data from the subset based on the number of observations
sample_data <- diamonds2[sample(1:nrow(diamonds2), num_obs,
                               replace=FALSE),]

# Create new dataframe with sampled data
diamonds3 <- sample_data

# View sampled dataset 
head(diamonds3)


##### DATA MODELING AND RESULTS #####

##### Data partitioning for machine learning #####

## Create `train` and `test` sets for `diamonds3` dataframe

set.seed(2021, sample.kind = "Rounding") # using R 3.5 or earlier, use `set.seed(1)`

# Split data into train and test sets
test_index <- createDataPartition(y = diamonds3$fprice, times = 1, p = 0.8, list = FALSE)
test_set <- diamonds3[test_index, ]
train_set <- diamonds3[-test_index, ]


##### Model 1: Decision Tree #####

# Fit the ml model
dtree_fit <- train(fprice ~., method = "rpart", data = train_set)

# Predictions on the train set
pred_train <- predict(dtree_fit, train_set)

# Write the confusion matrix to see the accuracy, 
# sensitivity and specificity of the predicted train set
print(confusionMatrix(pred_train, train_set$fprice))

# Predictions on the test set
pred_test <- predict(dtree_fit, test_set)

# Write the confusion matrix to see the accuracy, 
# sensitivity and specificity of the predicted test set
print(confusionMatrix(pred_test, test_set$fprice))


##### Model 2: Random Forest #####

# Fit the ml model
rforest_fit <- randomForest(fprice ~. , data = train_set,
                       importance = TRUE, ntrees = 10)

# Predictions on the train set
pred_train2 <- predict(rforest_fit, train_set)

# Write the confusion matrix to see the accuracy, 
# sensitivity and specificity of the predicted train set
print(confusionMatrix(pred_train2, train_set$fprice))

# Predictions on the test set
pred_test2 <- predict(rforest_fit, test_set)

# Write the confusion matrix to see the accuracy, 
# sensitivity and specificity of the predicted test set
print(confusionMatrix(pred_test2, test_set$fprice))


##### Addendum* #####

##### Model 3: Neural Network #####

# Before creating nn we should normalize our data using `min-max normalization`
normalize <- function(x){
  (x- min(x)) /(max(x)-min(x))
}

diamonds4 <-  diamonds3 %>% select_if(is.numeric)

# Normalize diamonds data from `diamonds4` dataset
# Set the `diamonds5` name to the normalized dataframe
diamonds5 <- as.data.frame(lapply(diamonds4, normalize))

# Check the first six rows of the updated data
head(diamonds5)

# Data partitioning for neural networks
test_index_n <- sample(nrow(diamonds5), 0.8 * nrow(diamonds5))
train_set_n <- diamonds5[test_index_n, ]
test_set_n <- diamonds5[-test_index_n, ]
nrow(train_set_n)
nrow(test_set_n)

# Create nn models
# Assign a variable `nn` for the `train_set_n`
# Please, wait... The process may take several minutes
# It depends of the computer resources 
nn <- neuralnet(price~carat+cut.Fair+cut.Good+cut.Ideal+cut.Premium+cut.Very_Good+
                  clarity.I1+clarity.IF+clarity.SI1+clarity.SI2+clarity.VS1+
                  clarity.VS2+clarity.VVS1+clarity.VVS2+
                  color.D+color.E+color.F+color.G+
                  color.H+color.I+color.J,
                data = train_set_n, hidden = c(5, 3),
                linear.output = TRUE)

# Assign a variable `nn2` for the `test_set_n`
nn2 <- neuralnet(price~carat+cut.Fair+cut.Good+cut.Ideal+cut.Premium+cut.Very_Good+
                   clarity.I1+clarity.IF+clarity.SI1+clarity.SI2+clarity.VS1+
                   clarity.VS2+clarity.VVS1+clarity.VVS2+
                   color.D+color.E+color.F+color.G+
                   color.H+color.I+color.J,
                 data = test_set_n, hidden = c(5, 3), 
                 linear.output = TRUE)

# Plot both neural networks
plot(nn)
plot(nn2)

# Predictions for training set
predict_train <- compute(nn, train_set_n)

# Predictions for testing set
predict_test <- compute(nn2, test_set_n)


##### Model Validation #####

# Results:
# Actual and prediction results for the train set 
results_train <- data.frame(actual = train_set_n$price, 
                            prediction = predict_train$net.result)
head(results_train)

# Actual and prediction results for the test set
results_test <- data.frame(actual = test_set_n$price,
                           prediction = predict_test$net.result)
head(results_test)


# Test the accuracy of our model 
predicted <- results_test$prediction * abs(diff(range(test_set_n$price))) + 
  min(test_set_n$price)
actual <- results_test$actual * abs(diff(range(test_set_n$price))) + min(test_set_n$price)
comparison <- data.frame(predicted, actual)
deviation <- ((actual - predicted)/actual)
comparison <- data.frame(predicted, actual, deviation)
accuracy <- 1 - abs(mean(deviation))

accuracy

##### Our accuracy of price prediction using neural network is 99% #####


