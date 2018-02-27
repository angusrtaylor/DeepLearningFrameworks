#install.packages("R.matlab")

process_cifar_mat <- function() {
  
  require(R.matlab)
  
  train_labels <- list()
  train_data <- list()
  for (i in seq(5)) {
    train <- readMat(paste0('./cifar-10-batches-mat/data_batch_', i, '.mat'))
    train_data[[i]] <- train$data
    train_labels[[i]] <- train$labels
  }
  
  x_train <- do.call(rbind, train_data)
  y_train <- do.call(rbind, train_labels)
  dim(x_train) <- c(50000, 3, 32, 32)
  
  test <- readMat('./cifar-10-batches-mat/test_batch.mat')
  
  x_test <- test$data
  y_test <- test$labels
  dim(x_test) <- c(10000, 3, 32, 32)
  
  rm(train_data, train_labels, train, test)
  
  list(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
  
}