# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:35:13 2016

@author: mavezsinghdabas
"""

import numpy as np 
import time

# ----------------------------------------------------------------------------------------

def loss_grad_svm_vectorized(W, X, y, reg):
    """
    Compute the loss and gradients using softmax function 
    with loop, which is slow.

    Parameters
    ----------
    W: (K, D) array of weights, K is the number of classes and D is the dimension of one sample.
    X: (D, N) array of training data, each column is a training sample with D-dimension.
    y: (N, ) 1-dimension array of target data with length N with lables 0,1, ... K-1, for K classes
    reg: (float) regularization strength for optimization.

    Returns
    -------
    a tuple of two items (loss, grad)
    loss: (float)
    grad: (K, D) with respect to W
    """

    dW = np.zeros(W.shape)
    loss = 0.0
    delta = 1.0

    num_train = y.shape[0]

    # compute all scores
    scores_mat = W.dot(X) # [C x N] matrix
    print(scores_mat.shape)    
    
    # get the correct class score 
    correct_class_score = scores_mat[y, range(num_train)] # [1 x N]
    
    margins_mat = scores_mat - correct_class_score + delta # [C x N]

    # set the negative score to be 0
    margins_mat = np.maximum(0, margins_mat)
    margins_mat[y, xrange(num_train)] = 0

    loss = np.sum(margins_mat) / num_train

    # add regularization to loss
    loss += 0.5 * reg * np.sum(W * W)

    # compute gradient
    scores_mat_grad = np.zeros(scores_mat.shape)

    # compute the number of margin > 0 for each sample
    num_pos = np.sum(margins_mat > 0, axis=0)
    scores_mat_grad[margins_mat > 0] = 1
    scores_mat_grad[y, xrange(num_train)] = -1 * num_pos

    # compute dW
    dW = scores_mat_grad.dot(X.T) / num_train + reg * W
    
    return loss, dW

# ----------------------------------------------------------------------------------------

def executeSVM(X_train, y_train, X_test, y_test):
    X_val = X_train[:1000, :] # take first 1000 for validation
    y_val = y_train[:1000]
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)    
    print(y_test.shape)

    W = np.random.randn(10, X_train.shape[0]) * 0.001
    print(W)
    print(y_train.shape)
    tic = time.time()
    loss_vec, grad_vect = loss_grad_svm_vectorized(W, X_train, y_train, 0)
    toc = time.time()
    print 'Vectorized loss: %f, and gradient: computed in %fs' % (loss_vec, toc - tic)




