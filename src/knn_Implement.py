# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:35:13 2016

@author: mavezsinghdabas
"""
import os
os.chdir(os.getcwd())
from LoadDataset import *
from collections import defaultdict


# This will be knn implementation for the Cifar dataset.
"""
We will be using the basic nearest neighbour classifier.
This is not a conventional method for image classifciation but will 
help in getting an idea for inage classigication.
"""

def getClosest(value):
    d = defaultdict(int)
    for i in value:
        d[i] += 1
    result = max(d.iteritems(),key =lambda x:x[1])
    return result[0]


# 1.  We will create a NearestNeighbour classifier. 
class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X, y_train ,k):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    knear = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L2 distance (sum of absolute value differences)
      distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
#      print("The Distance of ",i,distances)
#      min_index = np.argmin(distances) # get the index with smallest distance
      min_index = distances.argsort()[:k]
#      print(min_index)
#      print(y_train[min_index])
      knear = getClosest(y_train[min_index])
#      print(knear)
#      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
#      print(Ypred[i])
#      print(knear[i])
#      print("The iteration is =  %d",i)        
#      print("The predicted Y after iteration is",knear)

    return knear


# 2. Implementing the nearest neighbout classifier 


def executeNN(X_train, y_train, X_test, y_test):
#     Takign a subset to calculate the k minimum distance    
    X_train = X_train[:1000,:]
    y_train = y_train[:1000]
    X_test = X_test[:1000,:]
    y_test = y_test[:1000]
    
    nn = NearestNeighbor() # create a Nearest Neighbor classifier class
    print("Nearest Neighbor crated")
    nn.train(X_train,y_train ) # train the classifier on the training images and labels
    print("Y_test prediction started")    
    Yte_predict = nn.predict(X_test,y_train,8) # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    print("Calculating the accuracy")    
    print 'accuracy: %f' % ( np.mean(Yte_predict == y_test) )



# 3. Now we shall be implenenting knn. Along with cross Validation.


def executeKNN(X_train, y_train, X_test, y_test):
#    X_train = X_train[:1000,:]
#    y_train = y_train[:1000]
#    X_test = X_test[:1000,:]
#    y_test = y_test[:1000]
    
    # assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
    # recall Xtr_rows is 50,000 x 3072 matrix
    Xval_rows = X_train[:1000, :] # take first 1000 for validation
    Yval = y_train[:1000]
    Xtr_rows = X_train[1000:, :] # keep last 49,000 for train
    Ytr = y_train[1000:]
    
    # find hyperparameters that work best on the validation set
    validation_accuracies = []
    for k in [105]:
    
      # use a particular value of k and evaluation on validation data
      nn = NearestNeighbor()
      nn.train(Xtr_rows, Ytr)
      # here we assume a modified NearestNeighbor class that can take a k as input
      Yval_predict = nn.predict(Xval_rows, y_train,k = k)
      # print(Yval_predict)
      acc = np.mean(Yval_predict == Yval)
      print 'accuracy: %f' % (acc,)
    
      # keep track of what works on the validation set
      validation_accuracies.append((k,acc))















