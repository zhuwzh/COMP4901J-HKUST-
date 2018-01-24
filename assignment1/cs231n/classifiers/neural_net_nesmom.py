from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet_NesMom(object):
 
  
  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    loss function returns the loss with current update parameter (i.e. self.para)
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1'] 
    # sf.pa['W1'],sf.pa['b1'] is a tuple, while W1,b1 are scalers
    W2, b2 = self.params['W2'], self.params['b2'] 
    N, D = X.shape
    H, C = W2.shape

    # forward pass
    hidden_layer = np.maximum(0,np.dot(X,W1) + b1) # N * H
    scores = np.dot(hidden_layer,W2) + b2
    probs = np.exp(scores) / np.sum( np.exp(scores),axis=1,keepdims=True)

    # softmax loss with regularization 
    loss = -np.sum(np.log(probs[range(N),y]))
    loss /= N
    loss += reg * (np.sum(W1*W1) + np.sum(W2*W2))

    return loss

  def gradient(self,X,y,Wlist,reg):
    '''
    Compute the gradient of loss function at a specific point (i.e. Wdic)
    Input: Wdic is a list with elements in order : W1,b1,W2,b2
    Output: grads is a dictionary with keys: W1,b1,W2,b2
    '''
    W1, b1, W2, b2 = Wlist 
    N, D = X.shape
    H, C = W2.shape
   
    # forward pass
    hidden_layer = np.maximum(0,np.dot(X,W1) + b1) # N * H
    scores = np.dot(hidden_layer,W2) + b2
    probs = np.exp(scores) / np.sum( np.exp(scores),axis=1,keepdims=True)
   
    # backward propogation
    grads = {}

    temp = np.zeros((N,C))
    temp[range(N),y] = -1
    dscores = probs + temp # N * C
    
    dW2 = np.dot(hidden_layer.T, dscores)# H * C
    db2 = np.dot(np.ones((N,1)).T,dscores).squeeze() #(C,)
    
    dhidden_o = np.dot(dscores, W2.T) # N * H
    dhidden_i = (hidden_layer > 0) * dhidden_o # N * H
    
    dW1 = np.dot(X.T,dhidden_i) # D * H
    db1 = np.dot(np.ones((N,1)).T,dhidden_i).squeeze() # (H,)
    
    dW2 /= N
    dW2 += 2 * reg * W2
    grads['W2'] = dW2
    dW1 /= N
    dW1 += 2 * reg * W1
    grads['W1'] = dW1
    db2 /= N
    grads['b2'] = db2
    db1 /= N
    grads['b1'] = db1
    
    return grads
    

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100, mu=.9,
            batch_size=200, verbose=False):
    """
    Train this neural network using Stochastic Nesterov Momentumn .
    """
    
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    v_W1 = np.zeros_like(self.params['W1'])
    v_b1 = np.zeros_like(self.params['b1'])
    v_W2 = np.zeros_like(self.params['W2'])
    v_b2 = np.zeros_like(self.params['b2'])

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      
      # create a batch for this iteration
      idx_batch = np.random.choice(range(num_train),size=batch_size)
      X_batch = X[idx_batch,]
      y_batch = y[idx_batch]

      # Compute loss and gradients using the current minibatch
      loss = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)
      
      # Update self parameters by Stochastic Nesterov Momentumn
      W1_t = self.params['W1'] + mu * v_W1
      b1_t = self.params['b1'] + mu * v_b1
      W2_t = self.params['W2'] + mu * v_W2
      b2_t = self.params['b2'] + mu * v_b2
      
      Wlist = [W1_t,b1_t,W2_t,b2_t]
      grads_ahead = self.gradient(X=X_batch, y=y_batch, Wlist=Wlist, reg=reg)
      v_W1 = mu * v_W1 - learning_rate * grads_ahead['W1']
      v_b1 = mu * v_b1 - learning_rate * grads_ahead['b1']
      v_W2 = mu * v_W2 - learning_rate * grads_ahead['W2']
      v_b2 = mu * v_b2 - learning_rate * grads_ahead['b2']
      
      self.params['W1'] += v_W1
      self.params['b1'] += v_b1
      self.params['W2'] += v_W2
      self.params['b2'] += v_b2
 
      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):

    hidden_layer = np.maximum(0, np.dot(X,self.params['W1']) + self.params['b1'])
    scores = np.dot(hidden_layer,self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores,axis=1)

    return y_pred


