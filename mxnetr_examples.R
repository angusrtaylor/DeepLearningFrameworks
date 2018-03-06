.libPaths(c(.libPaths(), "/data/mlserver/9.2.1/libraries/RServer"))
require(mxnet)

# boston housing ----------------------------------------------------------

# from https://mxnet.incubator.apache.org/tutorials/r/fiveMinutesNeuralNetwork.html

require(mlbench)

data(BostonHousing, package="mlbench")

train.ind = seq(1, 506, 3)
train.x = data.matrix(BostonHousing[train.ind, -14])
train.y = BostonHousing[train.ind, 14]
test.x = data.matrix(BostonHousing[-train.ind, -14])
test.y = BostonHousing[-train.ind, 14]

# Define the input data
data <- mx.symbol.Variable("data")
# A fully connected hidden layer
# data: input source
# num_hidden: number of neurons in this hidden layer
fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)

# Use linear regression for the output layer
lro <- mx.symbol.LinearRegressionOutput(fc1)


mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=mx.cpu(),     num.round=50, array.batch.size=20,
                                     learning.rate=2e-6, momentum=0.9,  eval.metric=mx.metric.rmse)

preds = predict(model, test.x)

preds


# Kaggle handwritten digits -----------------------------------------------

# from https://mxnet.incubator.apache.org/tutorials/r/mnistCompetition.html

train <- read.csv('data/digits_train.csv', header=TRUE)
test <- read.csv('data/digits_test.csv', header=TRUE)

train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[,-1]
train.y <- train[,1]

train.x <- t(train.x/255)
test <- t(test/255)

dim(test)
dim(train.x)

table(train.y)

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

devices <- mx.cpu()
arr_iter <- mx.io.arrayiter(train.x, train.y,batch.size = 100, shuffle = TRUE)

mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=arr_iter,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.0007, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))


