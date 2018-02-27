# install.packages("reticulate", repos = "https://www.stats.bris.ac.uk/R/")
.libPaths(c(.libPaths(), "/data/mlserver/9.2.1/libraries/RServer"))

source('./common/process_cifar_mat.R')

cifar <- process_cifar_mat()
x_train <- cifar$x_train
y_train <- cifar$y_train
x_test <- cifar$x_test
y_test <- cifar$y_test
rm(cifar)


library(reticulate)

library(mxnet)

x_test <- mx.nd.array(src.array = x_test)

