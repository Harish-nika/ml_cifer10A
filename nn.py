import matplotlib.pyplot as plt
import numpy as np

class NeuralNetClassifier(object):
    def __init__(self):
        self.step_size=.001
        self.reg =1e-3
        pass

    def initialize_weights(self,d,k):   #dimensions,classes
        W = np.random.randn(d, k)         #multiply by "np.sqrt(2.0/d)" for  Calibrating the variances with 1/sqrt(n).
        # W[d-1]*=0 #make all bias terms 0
        # W=np.array([[ 0.03151032, -1.0425928 ],
        #             [0,                     0]] )
        # W=np.array([[ 1.30848596, 0.07582354,0.42190731],
        #             [-0.86197276,-0.67208923,0.61224682],
        #             [-0.        , 0.        ,0.        ]])

        return W

    def softmax(self, scores):
        # For single example and 3 classes: f is [123, 456, 789]
        # p = np.exp(f) / np.sum(np.exp(f))  # Bad: Numeric problem, potential blowup
        # instead: first shift the values of f so that the highest number is 0: f becomes [-666, -333, 0]

        scores -= scores.max(axis=1, keepdims=1)                 # left shift recommended to avoid overflow
        exp_scores = np.exp(scores)
        p = exp_scores / np.sum(exp_scores, axis=1, keepdims=1)  # safe to do now, gives the correct answer
        return p                                                 # p: NxK;

    def derivate_cross_entropy_loss(self, p, y, n):
        dscores = p
        dscores[range(n), y.flatten()] -= 1  # Note: grad becomes negative for correct class(as it should!)
        dscores /= n
        return dscores

  # """ X is N x D where each row is an example. Y is 1-dimension of size N """
    def train(self, X, y):
        # the nearest neighbor classifier simply remembers all the training data

        n,d=X.shape                 #number of examples, dimensions/features
        k=np.unique(y).shape[0]     #unique classes(0..k-1)
        h=100                       #size of hidden layer

        W1 = self.initialize_weights(d, h)
        b1 = np.zeros((1, h))
        W2 = self.initialize_weights(h, k)
        b2 = np.zeros((1, k))

        #GRADIENT DESCENT
        loss_axis=[]
        loop_count=200

        for i in range(loop_count):

            # 1. Forward pass -----------------------------------------------------------------------------
            # evaluate class scores, [N x K]
            h1=np.maximum(0,np.dot(X,W1)+b1)                           #hidden layer-1. (maximum: Apply RELU)
            scores = np.dot(h1, W2)+b2                                 #output layer: some people  call it softmax directly(when objective/loss fn is softmax)

            # compute the class probabilities(softmax)
            p = self.softmax(scores)

            # compute the loss: average cross-entropy loss and regularization
            corect_logprobs = -np.log(p[range(n),y]+1e-5)                             #column vector
            data_loss = np.sum(corect_logprobs)/n
            reg_loss = (1/2)*self.reg*(np.sum(W1*W1)+np.sum(W2*W2))                   #square of each weight
            loss=data_loss+reg_loss
            loss_axis.append(loss)
            # print "loss_matrix: \n", loss_matrix, "\n"

            # 2. Backward pass: Compute gradients-----------------------------------------------------------------------------
            # compute the gradient on output(scores)
            dscores = self.derivate_cross_entropy_loss(p, y, n)

            # backpropate the gradient to the parameter - W2
            dW2 = np.dot(h1.T, dscores)
            db2 = np.sum(dscores,axis=0,keepdims=True)             #sum about elements of axis-0 (first sum = first(row1)+first(row2)..)

            # backpropate the gradient to the parameter - h1
            dh1 = np.dot(dscores, W2.T)
            dh1[h1 <= 0] = 0          # backprop the ReLU non-linearity

            # backpropate the gradient to W1
            dW1 = np.dot(X.T, dh1)
            db1 = np.sum(dh1, axis=0,keepdims=True)

            dW2 += self.reg * W2      # regularization gradients
            dW1 += self.reg * W1


            # 3. perform parameter update -----------------------------------------------------------------------------
            W2 += (self.step_size * (-dW2))
            b2 += (self.step_size * (-db2))
            W1 += (self.step_size * (-dW1))
            b1 += (self.step_size * (-db1))

        print loss_axis
        plt.plot(np.arange(loop_count), loss_axis, 'ro')
        plt.show()

        print "Final W1: \n",W1
        print "Final W2: \n", W2

        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

  # """ X is N x D where each row is an example we wish to predict label for """
    def predict(self, X):
        n, d = X.shape

        h1 = np.maximum(0,np.dot(X,self.W1)+self.b1)
        scores = np.dot(h1,self.W2) +self.b2
        output = np.argmax(scores,axis = 1)  #index of max class(along "cols") for each row (returns 1D array)
        return np.array(output,ndmin=2).T   #return column vector
