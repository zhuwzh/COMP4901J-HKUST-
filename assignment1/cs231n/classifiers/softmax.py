import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  Prob = np.zeros((X.shape[0],W.shape[1])) #N * C
  Score = np.zeros_like(Prob) #N * C
  dScore = np.zeros_like(Prob)


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  Score = np.dot(X,W)
  Norm = np.sum(np.exp(Score),axis=1) # (N,)
  
  for i in range(X.shape[0]):
      for j in range(W.shape[1]):
          Prob[i,j] = np.exp(Score[i,j]) / Norm[i]
          if j == y[i]:
              dScore[i,j] = Prob[i,j] - 1

          else:
              dScore[i,j] = Prob[i,j]
          
          for k in range(W.shape[0]):
              dW[k,j] += dScore[i,j] * X[i,k]     

      loss += -np.log(Prob[i,y[i]])

  loss /= X.shape[0]
  dW /= X.shape[0]
  
  loss += reg * np.sum( (W*W)[:W.shape[0]-1] )
  copyW = W
  copyW[W.shape[0]-1,] = 0
  dW += 2 * reg * copyW

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  Prob = np.zeros((X.shape[0],W.shape[1])) # N * C
  Score = np.zeros_like(Prob) # N * C
  dScore = np.zeros_like(Prob) # N * C
  
  Score = np.dot(X,W)
  Prob = np.exp(Score) / np.sum(np.exp(Score),axis=1,keepdims=True)
  Temp = np.zeros((X.shape[0],W.shape[1]))
  Temp[range(X.shape[0]),y] = 1
  dScore = Prob - Temp
  
  loss = -np.sum(np.log(Prob[range(X.shape[0]),y]))
  dW = np.dot(X.T,dScore)
  
  loss /= X.shape[0]
  dW /= X.shape[0]
  
  loss += reg * np.sum( (W*W)[:W.shape[0]-1] )
  copyW = W
  copyW[W.shape[0]-1,] = 0
  dW += 2 * reg * copyW
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

