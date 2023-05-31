library(Metrics)
library(caret)
library(relaimpo)
library(stringr)
setwd('/Users/user/Documents/r_studio_projects/heart_failure')
printw = function(thing){
  print(thing, quote=F)
}

set.seed(352)
# Loading the data
# data = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv')
data = read.csv('heart_failure_clinical_records_dataset.csv')
printw(head(data))

# checking the shape of the data
printw(dim(data))

# checking for na
any_missing_data =  any(is.na(data))
if (any_missing_data == TRUE){
  any_missing_data = 'True'
} else {
  any_missing_data = 'False'
}

printw(sprintf('any missing values?: %s', any_missing_data))

# checking if the classes are balanced
class_and_frequencies = table(data[, ncol(data)])
printw(class_and_frequencies)
# The classes are not balanced, hence I resort to reducing the number of instances for the more populous category
data_0 = data[data$DEATH_EVENT == 0, ]
data_1 = data[data$DEATH_EVENT == 1, ]

data_0 = data_0[sample(1:203, size=96), ]
printw(data_0)

data = rbind(data_0, data_1)
printw(table(data[, ncol(data)])) # the data is now balanced

# checking the data types of each feature
dtypes = function(data){
  dtype_vector = c()
  
  for(i in 1:length(colnames(data))){
    dtype_vector = c(dtype_vector, typeof(data[, i]))
  }
  dtype_frame = data.frame(dtype_vector)
  return(dtype_frame)
}

dtypes = dtypes(data)
printw(dtypes)

# numerical data
data_num = data[, -c(2, 4, 6, 10, 11)]


# obtaining plots for the variables
for (k in 1:(ncol(data_num)-1)){
  png(file=sprintf('plot_of_Death Event against_%s_.png', colnames(data_num)[k]),
      height=6, 
      width=6, 
      units='in', 
      res = 35
  )
  
  plot(data[, ncol(data)], data[, k],
       ylab=colnames(data_num)[k], 
       xlab='Death Event', 
       main= sprintf('plot of Death Event against %s', chartr('_', ' ', str_to_title(colnames(data_num)[k]))) )
  dev.off()
  
}

# # checking for correlation between features
# printw('checking for any correlation between numerical features')
printw(cor(data_num[, (1:7)]))
# Hence, there is little correlation between the independent variables, hence we can assume  no multicollinearity


# checking for correlation betwen the features and the target
print('correlation between the numerical features and target')
printw(cor(data_num[, ]))

# from the plots, it can be seen that there is a relationship between the variables
# and the event that the patient died.

# feature selection using relative importance using R^2 partitioned by averaging
# over orders, i.e is the R^2 contribution averaged over orderings among regressors
# e.g. # like in Lindemann, Merenda and Gold (1980, p.119ff))  or Chevan and Sutherland (1991)
linear_model = lm(DEATH_EVENT~., data=data)
relative_importance = calc.relimp(linear_model, type='lmg', rela=T)
relative_importance = data.frame(relative_importance=sort(relative_importance$lmg, decreasing=TRUE))

printw(sprintf('relative_importance relative_importance:'))
printw(relative_importance)

feature_names = rownames(relative_importance)
selected_features = c()
for (i in 1:nrow(relative_importance)){
  if (relative_importance[i, ] > 0.01){
    selected_features = append(selected_features, feature_names[i])
  }
}

# The models seem to do well when the last two of the selected variables are removed
data = data[, c(selected_features[1:3], 'DEATH_EVENT') ]
selected_features = selected_features[1:3]
printw(sprintf('selected features %s', selected_features))

# splitting to testing and training datasets
train_index = createDataPartition(y = data$DEATH_EVENT, p= 0.8, list = FALSE)

train_data = data[train_index,]
test_data = data[-train_index, ]

# categorizing the target variable
train_data[["DEATH_EVENT"]] = factor(train_data[["DEATH_EVENT"]])
test_data[['DEATH_EVENT']] = factor(test_data[['DEATH_EVENT']])

# Controlling computational nuances of the train() method.
control <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

# 1) K-Nearest Neighbour
printw("")
printw(rep('_', 50))
printw('K-Nearest Neighbour')
printw(rep('_', 50))

knn_model <- train(DEATH_EVENT ~., data = train_data, method = "knn",
                   trControl=control,
                   preProcess = c("center", "scale"),
                   tuneLength = 10)
printw(knn_model)
plot(knn_model)

# making prediction, obtaining model accuracy and confusion matrix
y_predicted = predict(knn_model, test_data)

confusion_matrix = confusionMatrix(y_predicted, test_data$DEATH_EVENT)
printw(confusion_matrix)

accuracy = accuracy(y_predicted, test_data[, ncol(test_data)])
printw(sprintf('Test accuracy = %f', accuracy))


# 2) Support Vector Machine
printw("")
printw(rep('_', 50))
printw('Support Vector Machine')
printw(rep('_', 50))

svm_model <- train(DEATH_EVENT ~., data = train_data, method = "svmLinear",
                   trControl=trctrl,
                   preProcess = c("center", "scale"),
                   tuneLength = 10)


printw(svm_model)

# making prediction, obtaining model accuracy and confusion matrix
y_predicted = predict(svm_model, test_data)

confusion_matrix = confusionMatrix(y_predicted, test_data$DEATH_EVENT)
printw(confusion_matrix)

accuracy = accuracy(y_predicted, test_data[, ncol(test_data)])
printw(sprintf('Test accuracy = %f', accuracy))


# 3) Logistic Regression
printw("")
printw(rep('_', 50))
printw('Logistic Regression')
printw(rep('_', 50))

logistic_model =  train(DEATH_EVENT ~., data = train_data, method = "glm",
                        trControl=trctrl,
                        preProcess = c("center", "scale"),
                        tuneLength = 10)

printw(logistic_model)


# making prediction, obtaining model accuracy and confusion matrix
y_predicted = predict(logistic_model, test_data)

confusion_matrix = confusionMatrix(y_predicted, test_data$DEATH_EVENT)
printw(confusion_matrix)

accuracy = accuracy(y_predicted, test_data[, ncol(test_data)])
printw(sprintf('Test accuracy = %f', accuracy))


# 4) Decision Tree
printw("")
printw(rep('_', 50))
printw('Decision Tree')
printw(rep('_', 50))

decision_tree_model =  train(DEATH_EVENT ~., data=train_data,
                             method = "rpart",
                             trControl=trctrl,
                             preProcess = c("center", "scale"),
                             tuneLength = 10)

printw(decision_tree_model)

# making prediction, obtaining model accuracy and confusion matrix
y_predicted = predict(decision_tree_model, test_data)

confusion_matrix = confusionMatrix(y_predicted, test_data$DEATH_EVENT)
printw(confusion_matrix)

accuracy = accuracy(y_predicted, test_data[, ncol(test_data)])
printw(sprintf('Test accuracy = %f', accuracy))


# Saving K-Nearest Neighbour Model (the best model)
saveRDS(knn_model, 'decisionTree.rda')

# saving the prediction
printw(y_predicted)
y_and_y_predicted = data.frame(y_predicted=y_predicted, actual_y = test_data$DEATH_EVENT)

printw(y_and_y_predicted)
write.csv(y_and_y_predicted, 'prediction.csv')

# Reading saved model and making predictions thereby
saved_knn_model = readRDS('decisionTree.rda')
printw(predict(saved_knn_model, test_data))
