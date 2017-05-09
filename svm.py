import matplotlib.pyplot as plt
import numpy as np

class SupportVecotMachine(object):
    def __init__(self):
        self.step_size=.01
        self.reg = 1e-3
        pass

    def initialize_weights(self,d,k):   #dimensions,classes
        W = np.random.randn(d, k) * np.sqrt(2.0/d)        #Calibrating the variances with 1/sqrt(n).
        W[d-1]*=0                                         #make all bias terms 0
        # W=np.array([[ 0.03151032, -1.0425928 ],
        #             [0,                     0]] )

        return W

    # ALso called hinge loss because it hinges the loss = current score for correct class.
    def svmLoss(self,scores,y,delta=10):       #predicted outupt, actual output
        n,k=scores.shape

        # This creates "list" of "1D np-arrays". [array([3, 3]), array([4, 4]), array([5, 5])]
        correct_class_score=[np.full(k,s[y[i]]) for i,s in enumerate(scores)]    #check if creating array is costly operation. otherwise just iterate

        # This(below) makes it 2D np-array
        # array([[3, 3],
        #       [4, 4],
        #       [5, 5]])
        # No need fo this. because "scores" is already 2D-array, scores - correct_class_score returns 2d-np.array
        # correct_class_score=np.array(correct_class_score)

        loss = np.maximum(0,scores - correct_class_score + delta)   #element wise subtraction and addition
        for i in range(n):
            loss[i][y[i]]=0
        # loss[range(n),y.flatten()]=0

        return loss             #NxK  loss for example across each class

    def analytical_gradient(self,loss_matrix,X,y):

        n,d=X.shape
        k= loss_matrix.shape[1]     #loss_matrix: nxk

        # Convert loss matrix into gradient wrt (wi != correct_class):
        for i in range(n):
            for j in range(k):
                if (loss_matrix[i][j]>0):
                    loss_matrix[i][j]=1

        # wi == correct_class (note there is negative sign)
        for i in range(n):
            loss_matrix[i][y[i]] = -np.sum(loss_matrix[i])
        # print loss_matrix

        dW=np.zeros((d,k))
        for i in range(n):
            xi=X[i].reshape((d,1))
            dw=loss_matrix[i].reshape((1,k))
            gradi=np.dot(xi,dw)                     #gradient for ith example
            dW+=gradi

        return dW/n   #because it is was over n examples

    def derivate_svm(self, loss_matrix, y, n):
        dscore = loss_matrix
        dscore[dscore > 0] = 1
        dscore[range(n), y.flatten()] = -np.sum(dscore, axis=1)  # Z[range(2),[1,3]]=[12,312]    => Z[1][1]=12, Z[2][3]=312
        dscore /= n                          # Divide by n: take average over n examples. It is important to scale W appropriately.
                                             # Technically this should be done in the next step: dW=np.dot(X.T,dscores)/n
        return dscore

  # """ X is N x D where each row is an example. Y is 1-dimension of size N """
    def train(self, X, y):
        # the nearest neighbor classifier simply remembers all the training data

        n,d=X.shape #examples, #dimensions/features
        k=np.unique(y).shape[0]    #unique classes(0..k-1)

        bias =np.ones((n,1),int)
        X = np.concatenate( (X, bias), axis=1)        #add one dimension for bias
        d+=1
        W = self.initialize_weights(d,k)

        #GRADIENT DESCENT
        yaxis=[]
        loop_count=500

        for i in range(loop_count):
            # 1. Forward pass -----------------------------------------------------------------------------
            scores = np.dot(X, W)  # numpy array multiplicaiton
            # print "\n---------------------------------------------------\n"
            # print "loop_count: ", i
            # print "X: \n", X, "\n"
            # print "W: \n", W, "\n"
            # print "scores: \n",scores,"\n"

            # Half Vectorized: # svm_loss = self.svmLoss(scores, y)
            correct_class_score = scores[range(n),y.flatten()].reshape(n,1)        # column vector. don't forget to "flatten y"
            delta =30                                                              # margin (1, 10,20, 30 more is working better)
            svm_loss = np.maximum(0, scores - correct_class_score + delta)         # NxK
            svm_loss[range(n), y.flatten()] = 0
            reg_loss = (1 / 2) * self.reg * np.sum(W * W)                          #square of each weight (1/2 is included so it gets cancelled on derivation)
            total_loss = np.sum(np.sum(svm_loss))  + reg_loss                   #over n examples
            yaxis.append(total_loss/n)
            # print "svm_loss: \n", svm_loss, "\n"

            # 2. Backward pass: Compute gradients-----------------------------------------------------------------------------
            # compute the gradient on output - d(scores)                           # Half Vectotrized: # grad = self.analytical_gradient(svm_loss, X,y)
            dscore = self.derivate_svm(svm_loss, y, n)

            dW = np.dot(X.T, dscore)
            dW += self.reg * W          #reg gradient

            # 3. perform parameter update -----------------------------------------------------------------------------
            W += (self.step_size * (-dW))

        plt.plot(np.arange(loop_count), yaxis, 'ro')
        print yaxis
        print W

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


    # def numerical_gradient(self,f,W):
        # centered difference formula:

        # grad=np.zeros((W.shape))
        # h=0.00001

        # grad = (f(X+h)+f(X-h))/2*h

        # return grad    #DxK

        # it = np.nditer(loss, flags=['multi_index'])
        # while not it.finished:

        # print "%d <%s>" % (it[0], it.multi_index),
        # i=it.multi_index
        # loss
        # it.iternext()