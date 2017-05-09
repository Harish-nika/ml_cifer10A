import matplotlib.pyplot as plt
import numpy as np

class SoftmaxClassifier(object):
    def __init__(self):
        self.step_size=.01
        self.reg =1e-3
        pass

    def initialize_weights(self,d,k):   #dimensions,classes
        W = np.random.randn(d, k) * np.sqrt(2.0/d)        #Calibrating the variances with 1/sqrt(n).
        W[d-1]*=0 #make all bias terms 0
        # W=np.array([[ 0.03151032, -1.0425928 ],
        #             [0,                     0]] )
        # W=np.array([[ 1.30848596, 0.07582354,0.42190731],
        #             [-0.86197276,-0.67208923,0.61224682],
        #             [-0.        , 0.        ,0.        ]])

        return W

    def softmaxLoss(self,p,y):       #predicted outupt, actual output
        n=y.shape[0]
        loss_vector=[]      #list/1D array
        for i in range(n):
            loss_vector.append(-np.log(p[y[i]]))

        return np.array(loss_vector,ndmin=2).T      #return column vector

    def analytical_gradient(self,p,X,y):

        n,d=X.shape
        k= p.shape[1]     # "probabilities": nxk

        # partial gradient on incorrect class(vector w(j)): Convert loss matrix into gradient wrt wi != correct_class:
            #nothing
        # partial gradient on correct class(vector w(yi)>):
        for i in range(n):
            p[i][y[i]]=p[i][y[i]]-1     #negative

        #Sum over all the examples:
        grad=np.zeros((d,k))    #same dimensions as W
        for i in range(n):
            xi=X[i].reshape((d,1))
            dw=p[i].reshape((1,k))
            gradi=np.dot(xi,dw)
            grad+=gradi

        return grad

    def softmax(self, scores):
        # For single example and 3 classes: f is [123, 456, 789]
        # p = np.exp(f) / np.sum(np.exp(f))  # Bad: Numeric problem, potential blowup
        # instead: first shift the values of f so that the highest number is 0: f becomes [-666, -333, 0]

        scores -= scores.max(axis=1, keepdims=1)  # left shift recommended to avoid overflow
        exp_scores = np.exp(scores)
        p = exp_scores / np.sum(exp_scores, axis=1, keepdims=1)  # safe to do now, gives the correct answer
        return p  # p: NxK;

    def derivate_cross_entropy_loss(self, p, y, n):
        dscores = p
        dscores[range(n), y.flatten()] -= 1  # Note gradient/partial_derivative on correct class(node) is negative.
                                             # So vector Xi gets "added" to this classifier weights(w)
        dscores /= n                         # Divide by n: take average over n examples. It is important to scale W appropriately.
                                             # Technically this should be done in the next step: dW=np.dot(X.T,dscores)/n
        return dscores


    # """ X is N x D where each row is an example. Y is 1-dimension of size N """
    def train(self, X, y):
        # the nearest neighbor classifier simply remembers all the training data

        n,d=X.shape #examples, #dimensions/features
        k=np.unique(y).shape[0]    #unique classes(0..k-1)

        bias =np.ones((n,1),int)
        X = np.concatenate( (X, bias), axis=1)        #add one dimension for bias
        d+=1
        W = self.initialize_weights(d,k)
        print "intial W: \n",W

        #GRADIENT DESCENT
        loss_axis=[]
        loop_count=500

        for i in range(loop_count):

            # 1. Forward pass -----------------------------------------------------------------------------
            # evaluate class scores, [N x K]
            scores = np.dot(X, W)

            # compute the class probabilities
            p = self.softmax(scores)

            # compute the loss: average cross-entropy loss and regularization
            corect_logprobs = -np.log( p[range(n),y] + 1e-5)           # half-vectorized: corect_logprobs=self.softmaxLoss(p,y)
            data_loss = np.sum(corect_logprobs)/n
            reg_loss = (1/2)*self.reg*np.sum(W*W)                      # square of each weight
            loss=data_loss+reg_loss
            loss_axis.append(loss)
            # print "loss_matrix: \n", loss_matrix, "\n"

            # 2. Backward pass: Compute gradients-----------------------------------------------------------------------------
            # compute the gradient on output - d(scores)        (Half-vectorized: grad = self.analytical_gradient(p, X,y))
            dscores = self.derivate_cross_entropy_loss(p, y, n)

            # backpropate the gradient to the parameter - W
            dW=np.dot(X.T,dscores)

            dW+=self.reg*W      # regularization gradient

            # 3. perform parameter update -----------------------------------------------------------------------------
            W += (self.step_size * (-dW))

        plt.plot(np.arange(loop_count), loss_axis, 'ro')
        print loss_axis
        print "Final W: \n",W

        plt.show()
        self.W=W    #Just remember the weights not data

  # """ X is N x D where each row is an example we wish to predict label for """
    def predict(self, X):
        n, d = X.shape
        bias =np.ones((n,1),int)
        X = np.concatenate( (X, bias), axis=1)

        all_pred = np.dot(X,self.W)
        output = np.argmax(all_pred,axis = 1)  #index of max class(along "cols") for each row (returns 1D array)
        return np.array(output,ndmin=2).T   #return column vector

# Tips
'''
f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
'''