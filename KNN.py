import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass

  # """ X is N x D where each row is an example. Y is 1-dimension of size N """
    def train(self, X, y):
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def L1_loss(self,test):
        diff = np.abs(self.Xtr - test)  # find pixel wise difference for given test case with each Training example
        # L1  = NxD matrix
        return np.sum(diff, axis=1)  # sum c0,c1,c2...for each row (sum across column). distances = Nx1 matrix

    def L2_loss(self,test):
        sq_dis = np.sum(np.square(self.Xtr - test),axis=1)
        return np.sqrt(sq_dis)

  # """ X is N x D where each row is an example we wish to predict label for """
    def predict(self, X,k):
        num_test = X.shape[0]       # number of examples N = rows in X

        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)      # lets make sure that the output type matches the input type

        # loop over all test rows
        for i in xrange(num_test):
          # find the nearest training image to the i'th test image
          # using the L1 distance (sum of absolute value differences)

            loss= self.L2_loss(X[i, :])

            min_index = np.argmin(loss)      # get the index with smallest distance

            Ypred[i] = self.ytr[min_index]        # predict the label of the nearest example

        return Ypred

