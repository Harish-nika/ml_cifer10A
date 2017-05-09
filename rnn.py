# Observations:
#
# randomly initialising weights
# Adding bias terms for each neuron> Great impact
# step size = 0.95 instead 0.1 > great impact
# hidden dimensions: 4(poor), 8(fine), 16(good)
# regularization? Not having much effect. But if you keep, self.reg = .0001 works fine. not greater than this.

import matplotlib.pyplot as plt
import numpy as np
import csv

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper

class RNNClassifier(object):
    def __init__(self):
        # print "haha"
        pass

    @run_once   #This restricts multiple instances/objects of class RNNClassifier to call train(). Only one class object can call train() method.
    def my_init(self, d,k):
        self.hidden_dim = 16
        self.step_size = 1
        self.reg = 0#1e-3
        h = self.hidden_dim  # size of hidden layer

        self.W1 = self.initialize_weights(d, h)
        self.Wh = self.initialize_weights(h, h)
        self.b1 = np.zeros((1, h))  # h-bias, 1 for each classifier

        self.W2 = self.initialize_weights(h, k)  # h-dimensions, k-classifiers (size of word2vec of output language)
        self.b2 = np.zeros((1, k))  # 1-bias: scalar
        # print "haha"

    #out of three initializations np.random.randn(d, k) is giving better/faster convergence than other techniques
    def initialize_weights(self,d,k):   #dimensions,classes
        return np.random.randn(d, k) #* np.sqrt(2.0/d)        #Calibrating the variances with 1/sqrt(n).
        # return 2*np.random.random((d,k)) - 1


# SGD: learn the weights after each example. X contains one example
    def train(self, X, y):
        n, d = X.shape          #there are n words(n timestamps) having d features each
        k = y.shape[1]

        self.my_init(d,k)       #called just once

        dW2 = np.zeros_like(self.W2)
        dWh = np.zeros_like(self.Wh)
        dW1 = np.zeros_like(self.W1)
        db2 = np.zeros_like(self.b2)
        db1 = np.zeros_like(self.b1)
        # print dW1, "\n", dW2, "\n",dWh,"\n"

        h = np.zeros((n+1, 1, self.hidden_dim))  #h[t] stores output hidden layer at time t(i.e., (t-1)th example). [[1,2,3...16]]

        digit = []
        output = []
        total_error = 0
        dX = []

        # 1. Forward pass -----------------------------------------------------------------------------
        for i in xrange(n):  # i= 0,1,2,..n-1    (w1 w2 w3)
            t=i+1
            Xi = np.array(X[i],ndmin=2)     #1xd
            yi = np.array(y[i],ndmin=2)     #1x1
            # print Xi,yi

            net1 = np.dot(Xi, self.W1) + np.dot(h[t-1], self.Wh) + self.b1       #h[0]=0. X[i]=1xd, W=dxh
            h[t] = 1/(1+np.exp(-net1))       #1xh

            net2 = np.dot(h[t], self.W2) + self.b2   #1xh, hx1 > 1x1
            scores = 1/(1+np.exp(-net2))     #sigmoid(-net)

            total_error += 0.5 * np.sum(np.square(yi - scores))
            digit.append(np.round(scores[0][0]))

            output.append(scores[0])


        scores = np.array(output)
        # 2. Backward pass: Compute gradients-----------------------------------------------------------------------------
        dnet1_next_layer = np.zeros((1, self.hidden_dim))
        for t in xrange(n,0,-1):  # t = d,d-1,d-2,..1
            i=t-1
            Xi = np.array(X[i], ndmin=2)
            yi = np.array(y[i], ndmin=2)  # 1x1

            dscore = -(yi - scores[i])
            dnet2 = dscore * (scores[i] * (1 - scores[i]))

            dW2 += np.dot(h[t].T, dnet2)                 # accumulate     hx1,1x1 > hx1
            db2 += np.sum(dnet2, axis=0, keepdims=True)  # accumulate
            dht = np.dot(dnet2, self.W2.T) + np.dot(dnet1_next_layer, self.Wh.T)   #From_output_layer + h[t]_was_also_input_to_next_hidden_layer

            dnet1 = dht * (h[t] * (1-h[t]))         # 1xh, 1xh

            dWh += np.dot(h[t - 1].T, dnet1)        # h[0] = 0
            dW1 += np.dot(Xi.T, dnet1)
            db1 += np.sum(dnet1, axis=0,keepdims=True)

            dX.append(np.dot(dnet1, self.W1.T)[0])  # 1xh, (dxh).T > 1xd

            dnet1_next_layer = dnet1

        # regularization gradient
        dW2 += self.reg * self.W2
        dWh += self.reg * self.Wh
        dW1 += self.reg * self.W1

        # 3. perform parameter update -----------------------------------------------------------------------------
        self.W2 += (self.step_size * (-dW2))
        self.b2 += (self.step_size * (-db2))
        self.Wh += (self.step_size * (-dWh))
        self.W1 += (self.step_size * (-dW1))
        self.b1 += (self.step_size * (-db1))

        return total_error

    def predict(self, X):
        n, d = X.shape  # 2x1
        h = np.zeros((n+1, self.hidden_dim))  #h[t] stores output hidden layer at time t(i.e., (t-1)th example).
        scores=[]
        # Feed-forward--------------------------------
        for i in xrange(n):  # i= 0,1,2,..n-1    (word0 word1 word2)
            t=i+1
            net1 = np.dot(X[i], self.W1) + np.dot(h[t-1], self.Wh) +self.b1       #h[0]=0. X[i]=1xd, W=dxh
            h[t] = 1/(1+np.exp(-net1))

            net2 = np.dot(h[t], self.W2)+self.b2
            scores.append(1/(1+np.exp(-net2)))   #sigmoid(-net)
        return np.array([scores]).T
#End Class RNNClassifier-----------------------------------

def prepare():
    f = open("/Users/nj/UFL/ML/hw2/foo.csv", 'rb')
    reader = csv.reader(f)
    split=150
    X,Y = [],[]

    for i,row in enumerate(reader):
        if(i==0):continue
        X.append([float(row[0]), float(row[1])])
        Y.append([float(row[2]), float(row[3])])

    X = np.array(X, ndmin=2)
    Y = np.array(X, ndmin=2)

    return X[:split], Y[:split], X[split:],Y[split:]   #y1[:split],y2[:split], X[split:],y1[split:],y2[split:]

def plot_graph(n, loss_axis):
    # print loss_axis
    plt.xlabel("# ith Example", fontsize=15)
    plt.ylabel("Training Error", fontsize=15)
    plt.plot(np.arange(n), loss_axis, '-')
    plt.show()

def TestScript1(model, X,Y, Xt,Yt):

    n=X.shape[0]
    loopcount=1
    loss_axis = []
    for k in xrange(loopcount):

        for i,xi in enumerate(X):   #train example by example.
            Xi = np.array(xi,ndmin=2).T     #Note Xi = one sentence = w1 w2 w2. where w1, w2, are word2vec representations
            Yi = np.array(Y[i],ndmin=2).T
            loss_axis.append(model.train(Xi, Yi))   #need to return gradient for each context vector

    plot_graph(n*loopcount, loss_axis)

    for i,xi in enumerate(Xt):   #cannot do anything but train example by example.
        Xi = np.array(xi,ndmin=2).T     #Note Xi = one sentence = w1 w2 w2. where w1, w2, are word2vec representations
        p = model.predict(Xi)

    # p1,p2 = model.predict(Xt)         # need to modify knn.predict to support k
    # print "y1 Mean,Std. deviation: ", np.mean(yt1), np.std(yt1)
    # print "y2 Mean,Std. deviation: ", np.mean(yt2), np.std(yt2)
    # print "Predicted y1 Mean,Std. deviation: ", np.mean(p1), np.std(p1)
    # print "Predicted y2 Mean,Std. deviation: ", np.mean(p2), np.std(p2)

def prepare2():
    rnn = RNNClassifier()
    int2binary = {}
    binary_dim = 8
    loss_axis = []
    loopcount=10000

    largest_number = pow(2, binary_dim)
    binary = np.unpackbits(
        np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

    for i in range(largest_number):
        int2binary[i] = binary[i]


    for i in xrange(loopcount):
        # generate a simple addition problem (a + b = c)
        a_int = np.random.randint(largest_number / 2)  # int version
        a = int2binary[a_int]  # binary encoding

        b_int = np.random.randint(largest_number / 2)  # int version
        b = int2binary[b_int]  # binary encoding

        # true answer
        c_int = a_int + b_int
        c = int2binary[c_int]

        # print a,b,c
        X = np.array(zip(a[::-1],b[::-1]))
        Y = np.array(c[::-1],ndmin=2).T
        # print X,"\n",Y

        # d = rnn.train(X,Y)    #calculated sum
        loss_axis.append(rnn.train(X,Y))
        # if i%1000==0: print c[::-1], "\n",d, "\n--------\n"
    plot_graph(loopcount, loss_axis)

# Main
def main():
    prepare2()
    # X,Y, Xt,Yt = prepare()     #Expects: X= float data_type, y = int data type
    # print len(X),len(Y),len(Xt),len(Yt)
    #
    # n, d = X.shape  # number of words, word2vec dimensions/features
    # k = Y.shape[1]  # word2vec dimension in output language
    #
    # n,d,k = 2, 1, 1
    # rnn = RNNClassifier(n,d,k)
    #
    # start_time = timeit.default_timer()
    # TestScript1(rnn, X,Y, Xt,Yt)
    # print "ExecutionTime: ", (timeit.default_timer() - start_time)
    # W=[[1,2],[3,5]]
    # np.savetxt("rnn", W)  # print "Final W: \n",W

main()