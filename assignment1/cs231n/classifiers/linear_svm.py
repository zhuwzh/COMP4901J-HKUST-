import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
       NOTICE: Last row of W is bias term.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
       NOTICE:Here we assume the last column of X is np.ones(N)
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero , D * C
  dS = np.zeros((X.shape[0],W.shape[1])) # N * C

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]
      for j in xrange(num_classes):
          if j == y[i]:
              continue # continue to next iteration. similar use for break func
          margin = scores[j] - correct_class_score + 1 # note delta = 1
          if margin > 0:                  
              loss += margin
              dS[i,j] = 1
              dS[i,y[i]] += -1
  dS /= num_train
  dW = np.dot(X.T,dS)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum( (W*W)[:W.shape[0]-1] )
  copyW = W
  copyW[W.shape[0]-1,] = 0
  dW += 2 * reg * copyW
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
#  W = W.astype('float128')
#  loss = np.array(0.0,dtype='float128')
#  dW = np.zeros(W.shape,dtype='float128') # initialize the gradient as zero
#  Score = np.zeros((X.shape[0],W.shape[1]),dtype='float128')

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  Score = np.zeros((X.shape[0],W.shape[1]))
#  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  Score = np.dot(X,W) # N * C
  correct_class_score = np.diagonal(Score[:,y])# (N,)
  ##correct_class_score = Score[range(Score.shape[0]),y]
  Margin = (Score.T - correct_class_score + 1).T # N * C
  loss = np.sum( np.maximum(Margin, 0) ) - X.shape[0]
  loss /= X.shape[0]
  loss += reg * np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  dS = np.ones((X.shape[0],W.shape[1])) # N * C
  Boo = (Margin > 0) # N * C 
  dS *= Boo # N * C
  count = np.sum(dS,axis = 1) - 1 # (N,)
  dS[range(dS.shape[0]),y] = -count
  
  dS /= X.shape[0]
  dW = np.dot(X.T,dS)
  dW += 2 * reg * W
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
