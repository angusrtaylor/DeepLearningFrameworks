# download matlab cifar data from https://www.cs.toronto.edu/~kriz/cifar.html

# install.packages("reticulate", repos = "https://www.stats.bris.ac.uk/R/")
.libPaths(c(.libPaths(), "/data/mlserver/9.2.1/libraries/RServer"))

library(reticulate)

library(mxnet)

source('./common/process_cifar_mat.R')

cifar <- process_cifar_mat()
x_train <- cifar$x_train
y_train <- cifar$y_train
x_test <- cifar$x_test
y_test <- cifar$y_test
rm(cifar)

source_python('./common/params.py')

create_symbol <- function(){
  
  data <- mx.symbol.Variable('data')
  # size = [(old-size - kernel + 2*padding)/stride]+1
  # if kernel = 3, pad with 1 either side
  conv1 <- mx.symbol.Convolution(data=data, num_filter=50, pad=c(1,1), kernel=c(3,3))
  relu1 <- mx.symbol.Activation(data=conv1, act_type="relu")
  conv2 <- mx.symbol.Convolution(data=relu1, num_filter=50, pad=c(1,1), kernel=c(3,3))
  pool1 <- mx.symbol.Pooling(data=conv2, pool_type="max", kernel=c(2,2), stride=c(2,2))
  relu2 <- mx.symbol.Activation(data=pool1, act_type="relu")
  drop1 <- mx.symbol.Dropout(data=relu2, p=0.25)
  
  conv3 <- mx.symbol.Convolution(data=drop1, num_filter=100, pad=c(1,1), kernel=c(3,3))
  relu3 <- mx.symbol.Activation(data=conv3, act_type="relu")
  conv4 <- mx.symbol.Convolution(data=relu3, num_filter=100, pad=c(1,1), kernel=c(3,3))
  pool2 <- mx.symbol.Pooling(data=conv4, pool_type="max", kernel=c(2,2), stride=c(2,2))
  relu4 <- mx.symbol.Activation(data=pool2, act_type="relu")
  drop2 <- mx.symbol.Dropout(data=relu4, p=0.25)
  
  flat1 <- mx.symbol.Flatten(data=drop2)
  fc1 <- mx.symbol.FullyConnected(data=flat1, num_hidden=512)
  relu7 <- mx.symbol.Activation(data=fc1, act_type="relu")
  drop4 <- mx.symbol.Dropout(data=relu7, p=0.5)
  fc2 <- mx.symbol.FullyConnected(data=drop4, num_hidden=N_CLASSES) 
  
  input_y <- mx.symbol.Variable('softmax_label')  
  mx.symbol.SoftmaxOutput(data=fc2, label=input_y, name="softmax")

}

sym <- create_symbol()

if (GPU) {
  ctx = mx.gpu(0)
} else {
  ctx = mx.cpu()
}
x_train_nd = aperm(x_train)
#y_train_nd = aperm(y_train)
y_train_nd <- y_train
dim(y_train_nd) <- 50000
train_iter = mx.io.arrayiter(data = x_train_nd, label = y_train_nd, batch.size = BATCHSIZE, shuffle = TRUE)

model <- mx.model.FeedForward.create(
  symbol = sym,
  X = train_iter,
  #y = y_train_nd,
  ctx = ctx,
  num.round = 3,
  array.batch.size = BATCHSIZE,
  learning.rate = LR,
  momentum = MOMENTUM,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.Xavier(rnd_type = 'uniform'),
  epoch.end.callback = mx.callback.log.train.metric(100)
)

preds <- predict(model, aperm(x_test))
