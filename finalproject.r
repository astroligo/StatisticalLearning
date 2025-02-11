                                    #########################################
                                     # STATISTICAL LEARNING: FINAL PROJECT #
                                               # READING SIGNS #
                                    #########################################
                                         
cat("\n#### STATISTICAL LEARNING: FINAL PROJECT ####\n")

# required libraries:
library(keras3)
library(caret)
library(tree)
library(randomForest)

###########################################################################################################
### TASK 1 : Build a Deep Feedforward Neural Network ###
###########################################################################################################
cat("\n## TASK 1 : Build a Deep Feedforward Neural Network ##\n")

## STEP 1: Preprocessing Data ##
# dataset source: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
# dataset should already be clean from dataset source description, 
# but an additional check is performed anyway in the preprocessing steps below
cat("\n# STEP 1: Preprocessing Data #\n")

# importing dataset (dowloaded already as pre-divided train&test datasets):
mnist_train <- read.csv("sign_mnist_train.csv", header = TRUE, sep = ",")
mnist_test <- read.csv("sign_mnist_test.csv", header = TRUE, sep = ",")

# displaying size of train and test dataset:
cat("Train Dataset dimensions: ",dim(mnist_train),"\n")
cat("Test Dataset dimensions: ",dim(mnist_test), "\n")

# overview of data:
#View(head(mnist_train, 5))
print(head(mnist_train[, 1:5], 5)) #displaying only first 5 rows & columns
#View(head(mnist_test, 5))
print(head(mnist_test[, 1:5], 5)) #displaying only first 5 rows & columns

# removing any rows with missing values:
na_row <- apply(mnist_train, 1, function(x){any(is.na(x))})
cat("Number of rows in training dataset with missing values: ",sum(na_row ),"\n")
train_filtered <- mnist_train[!na_row ,]
na_row  <- apply(mnist_test, 1, function(x){any(is.na(x))})
cat("Number of rows in testing dataset with missing values: ",sum(na_row ),"\n")
test_filtered <- mnist_test[!na_row ,]

# removing any rows with outlier values (pixel values should be between 0 and 255)
outlier_row <- apply(mnist_train, 1, function(x){any(x>255 | x<0)})
cat("Number of rows in training dataset with outlier values: ",sum(outlier_row),"\n")
train_filtered <- mnist_train[!outlier_row,]
outlier_row <- apply(mnist_test, 1, function(x){any(x>255 | x<0)})
cat("Number of rows in training dataset with outlier values: ",sum(outlier_row),"\n")
test_filtered <- mnist_test[!outlier_row,]

# checking if both train and test datasets are complete (0-25 letters/labels)
cat("All training set labels: ",sort(unique(train_filtered$label)),"\n")
cat("All testing set labels: ",sort(unique(test_filtered$label)),"\n")

# checking class/labels distributions (checking for any class imbalance to choose an appropriate model)
png(file = "classlabels_distribution.png",width = 2000, height = 900)
par(mfrow = c(1, 2))
train_counts <- table(train_filtered$label)
test_counts <- table(test_filtered$label)
barplot_train <- barplot(train_counts,main="Training Set Class Distribution",xlab="Label (=Letter)",
  ylab="Frequency",xaxt = "n",ylim=c(0, max(train_counts) + 1),cex.lab = 1.5,cex.main = 2)
text(x = barplot_train, y = 0, labels = LETTERS[-c(10, 26)],pos = 3,cex = 2, col = "black")  
barplot_test <- barplot(test_counts,main="Testing Set Class Distribution",xlab="Label (=Letter)",
  ylab="Frequency",xaxt = "n",ylim=c(0, max(test_counts) + 1),cex.lab = 1.5,cex.main = 2)
text(x = barplot_test, y = 0, labels = LETTERS[-c(10, 26)],pos = 3,cex = 2, col = "black") 
dev.off()

# scaling pixel values (Neural Network sensitive to scaling) + transforming to matrix for neural network input
x_train <- as.matrix(train_filtered[,2:ncol(train_filtered)]/255)
x_test <- as.matrix(test_filtered[,2:ncol(test_filtered)]/255)
#View(head(x_train, 4))
print(head(x_train[, 1:5], 5))

# transforming output column (labels) to one-hot encoding for Neural Network training:
y_train <- to_categorical(train_filtered$label)
y_test <- to_categorical(test_filtered$label)
#View(head(y_train, 4))
print(head(y_train[, 1:5], 5))

## STEP 2: Building the Model (Deep Feedforward Neural Network) ##
# based on keras neural network example values
cat("\n# STEP 2: Building the Model #\n")

deep_nn <- keras_model_sequential(input_shape = c(784))
deep_nn |> 
  layer_dense(units = 256, activation = 'relu') |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 128, activation = 'relu') |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 25, activation = 'softmax') #one unit per class
summary(deep_nn)

deep_nn |> compile(loss = 'categorical_crossentropy',optimizer = optimizer_rmsprop(),metrics = c('accuracy')) #categorical crossentropy == multi-class logistic regression loss function


## STEP 3: Training and Testing the Model ##
# fitting model:
start <- Sys.time()
history <- deep_nn |> fit(
  x_train, y_train,
  epochs = 50, batch_size = 128,
  validation_split = 0.2
) 
end <- Sys.time()
cat("\nNeural Network Training Time (sec): ",end-start,"\n")

# visualizations:
# training & validation accuracy
png(file="neuralnetwork_accuracy.png")
plot(history$metrics$accuracy,type="o",col = "red", lwd = 2, xlab = "Epochs", ylab = "Accuracy", 
     main = "Training and Validation Accuracy",ylim=c(0,1))
lines(history$metrics$val_accuracy, col = "orange", lwd = 2,type="o")
legend("topleft", legend = c("Training Accuracy", "Validation Accuracy"), 
       col = c("red", "orange"), lwd = 2)
dev.off()

# Neural Network Evaluation:
# training & validation loss
png(file="neuralnetwork_loss.png")
plot(history$metrics$loss, type="o",col = "blue", lwd = 2, xlab = "Epochs", ylab = "Loss", 
     main = "Training and Validation Loss",ylim=c(0,3))
lines(history$metrics$val_loss, col = "lightblue", lwd = 2,type="o")
legend("topright", legend = c("Training Loss", "Validation Loss"), 
       col = c("blue", "lightblue"), lwd = 2)
dev.off()

# test accuracy
test_performance <- deep_nn %>% evaluate(x_test, y_test)
test_accuracy <- test_performance$accuracy
cat("\nTest accuracy: ", test_accuracy,"\n")

# confusion matrix
nn_pred_probs <- predict(deep_nn, x_test)  
nn_pred <- apply(nn_pred_probs, 1, which.max) - 1
nn_cm <- confusionMatrix(as.factor(nn_pred), as.factor(test_filtered$label))

# precision, recall, and F1-score
nn_precision <- nn_cm$byClass[, "Precision"]
nn_recall <- nn_cm$byClass[, "Recall"]
nn_f1 <- nn_cm$byClass[, "F1"]
cat("\nNeural Network - Precision (per class): \n", nn_precision, "\n")
cat("Neural Network - Recall (per class): \n", nn_recall, "\n")
cat("Neural Network - F1-Score (per class): \n", nn_f1, "\n")
cat("\nNeural Network - Average Precision: \n", mean(nn_precision), "\n")
cat("Neural Network - Average Recall: \n", mean(nn_recall), "\n")
cat("Neural Network - Average F1-Score: \n", mean(nn_f1), "\n")


###########################################################################################################
## TASK 2: Baseline Comparison with Traditional SL Models ##
###########################################################################################################

set.seed(123) #for reproducibility

## STEP 1: Building the Model (Random Forest) ##

# First try-out Model: simple Decision Tree Classifier
decision_tree <- tree(as.factor(train_filtered$label) ~., data = train_filtered)
predictions <- predict(decision_tree, newdata = train_filtered, type = "class")
summary(decision_tree)
confusion_matrix <- table(Predicted = predictions, Actual = train_filtered$label)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("\n","Decision Tree Accuracy:", accuracy, "\n")
png(file = "decision_tree_classification.png")
plot(decision_tree)
text(decision_tree, pretty = 0)
dev.off()
# --> as expected very low accuracy and not able to predict most classes (class imbalance in data)
# trying Random Forest Classifier instead

# Second Model: Random Forest Classifier
# tuning hyperparameter mtry
tuned_rf <- tuneRF(train_filtered[, -which(names(train_filtered) == "label")], as.factor(train_filtered$label), stepFactor = 1.5,
    improve = 0.01,trace = TRUE)  

# training model with optimal hyperparameter mtry
start <- Sys.time()
rf_model <- randomForest(as.factor(train_filtered$label) ~ ., data = train_filtered, ntree = 50, mtry = tuned_rf[which.min(tuned_rf[, 2]), 1], 
    nodesize = 1)
end <- Sys.time()
cat("\nRandom Forest Training Time (sec): ", end - start)


## STEP 2: Evaluating the Model ##

# Evaluating Model:
# plotting error vs. number of trees
png(file = "RandomForest_numtrees.png")
plot(rf_model,main = "Random Forest Classification Model")
dev.off()

# accuracy
rf_pred <- predict(rf_model, newdata = test_filtered)
rf_accuracy <- mean(rf_pred == test_filtered$label)
cat("\nRandom Forest Accuracy after Tuning: ", rf_accuracy)

# confusion matrix
rf_pred_probs <- predict(rf_model, newdata = test_filtered, type = "prob")
rf_pred <- predict(rf_model, newdata = test_filtered, type = "response")
rf_cm <- confusionMatrix(as.factor(rf_pred), as.factor(test_filtered$label))
cat("\nRandom Forest Confusion Matrix:\n")
print(rf_cm)

# precision, Recall, F1-Score
rf_precision <- rf_cm$byClass[, "Precision"]
rf_recall <- rf_cm$byClass[, "Recall"]
rf_f1 <- rf_cm$byClass[, "F1"]
cat("Random Forest - Precision (per class): ", rf_precision, "\n")
cat("Random Forest - Recall (per class): ", rf_recall, "\n")
cat("Random Forest - F1-Score (per class): ", rf_f1, "\n")
cat("Random Forest - Average Precision: ", mean(rf_precision), "\n")
cat("Random Forest - Average Recall: ", mean(rf_recall), "\n")
cat("Random Forest - Average F1-Score: ", mean(rf_f1), "\n")

###########################################################################################################

cat("\n########### End of Script ##########\n")
