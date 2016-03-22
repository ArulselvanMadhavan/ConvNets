# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:35:13 2016

@author: mavezsinghdabas
"""

import numpy as np 
import time
from CrossValidation import *
from linear_classfier import *

# ----------------------------------------------------------------------------------------

def executeSVM(X_train, y_train, X_test, y_test):
    learning_rates = [1e-5, 1e-8]
    regularization_strengths = [10e2, 10e4]
    results = {}
    best_val = -1
    best_svm = None
    # X_train = getCIFAR_as_32Pixels_Image(X_train)
    # OX_test = getCIFAR_as_32Pixels_Image(OX_test)
    accuracy = []
    totalAccuracy = 0.0

    ## Implementing Cross Validation
    crossValidObj = CrossValidation(5, X_train, y_train)
    foldsGen = crossValidObj.generateTrainAndTest()
    for i in xrange(5):
        next(foldsGen)
        X_test = OX_test
        X_train = crossValidObj.train
        y_train = crossValidObj.labels_train
        X_val = crossValidObj.test
        y_val = crossValidObj.labels_test

        # Preprocessing: reshape the image data into rows
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_val = np.reshape(X_val, (X_val.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))

        # Normalize the data: subtract the mean image
        mean_image = np.mean(X_train, axis = 0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

        # Add bias dimension and transform into columns
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
        X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T
        
        SVM_sgd = SVM()
        
        losses_sgd = SVM_sgd.train(X_train, y_train, method='sgd', batch_size=200, learning_rate=1e-6,
                      reg = 1e5, num_iters=1000, verbose=False, vectorized=True)

        
        y_train_pred_sgd = SVM_sgd.predict(X_train)[0]
        print 'Training accuracy: %f' % (np.mean(y_train == y_train_pred_sgd))
        y_val_pred_sgd = SVM_sgd.predict(X_val)[0]
        print 'Validation accuracy: %f' % (np.mean(y_val == y_val_pred_sgd))

        i = 0
        interval = 5
        for learning_rate in np.linspace(learning_rates[0], learning_rates[1], num=interval):
            i += 1
            print 'The current iteration is %d/%d' % (i, interval)
            for reg in np.linspace(regularization_strengths[0], regularization_strengths[1], num=interval):
                svm = SVM()
                svm.train(X_train, y_train, method='sgd', batch_size=200, learning_rate=learning_rate,
                      reg = reg, num_iters=1000, verbose=False, vectorized=True)
                y_train_pred = svm.predict(X_train)[0]
                y_val_pred = svm.predict(X_val)[0]
                train_accuracy = np.mean(y_train == y_train_pred)
                val_accuracy = np.mean(y_val == y_val_pred)
                results[(learning_rate, reg)] = (train_accuracy, val_accuracy)
                if val_accuracy > best_val:
                    best_val = val_accuracy
                    best_svm = svm
                else:
                    pass
        
        # Print out the results
        for learning_rate, reg in sorted(results):
            train_accuracy,val_accuracy = results[(learning_rate, reg)]
            print 'learning rate %e and regularization %e, \n \
            the training accuracy is: %f and validation accuracy is: %f.\n' % (learning_rate, reg, train_accuracy, val_accuracy)
            print accuracy
        
        
        y_test_predict_result = best_svm.predict(X_test)
        y_test_predict = y_test_predict_result[0]
        test_accuracy = np.mean(oy_test == y_test_predict)
        accuracy.append(test_accuracy)
        totalAccuracy+=test_accuracy
        print 'The test accuracy is: %f' % test_accuracy
    print accuracy
    avgAccuracy = totalAccuracy / 5.0
    print 'Average Accuracy: %f' % avgAccuracy

