

import time
from linear_classfier import *
from LoadDataset import *
from CrossValidation import *
import numpy as np
###########################################################################################
""" The Softmax Regression class """
def execute_softmax(X_train,y_train,OX_test,oy_test):

    learning_rates = [1e-5, 1e-8]
    regularization_strengths = [10e2, 10e4]
    results = {}
    best_val = -1
    best_softmax = None
    # X_train = getCIFAR_as_32Pixels_Image(X_train)
    # OX_test = getCIFAR_as_32Pixels_Image(OX_test)
    accuracy = []
    totalAccuracy = 0.0

    ## Implementing Cross Validation
    crossValidObj = CrossValidation(5, X_train, y_train)
    foldsGen = crossValidObj.generateTrainAndTest()
    for i in range(5):
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

        softmax_sgd = Softmax()
        tic = time.time()
        losses_sgd = softmax_sgd.train(X_train, y_train, method='sgd', batch_size=200, learning_rate=1e-6,
                      reg = 1e5, num_iters=1000, verbose=False, vectorized=True)
        toc = time.time()


        y_train_pred_sgd = softmax_sgd.predict(X_train)[0]
        print('Training accuracy: %f' % (np.mean(y_train == y_train_pred_sgd)))
        y_val_pred_sgd = softmax_sgd.predict(X_val)[0]
        print('Validation accuracy: %f' % (np.mean(y_val == y_val_pred_sgd)))


        # Choose the best hyperparameters by tuning on the validation set
        i = 0
        interval = 5
        for learning_rate in np.linspace(learning_rates[0], learning_rates[1], num=interval):
            i += 1
            print('The current iteration is %d/%d' % (i, interval))
            for reg in np.linspace(regularization_strengths[0], regularization_strengths[1], num=interval):
                softmax = Softmax()
                softmax.train(X_train, y_train, method='sgd', batch_size=200, learning_rate=learning_rate,
                      reg = reg, num_iters=1000, verbose=False, vectorized=True)
                y_train_pred = softmax.predict(X_train)[0]
                y_val_pred = softmax.predict(X_val)[0]
                train_accuracy = np.mean(y_train == y_train_pred)
                val_accuracy = np.mean(y_val == y_val_pred)
                results[(learning_rate, reg)] = (train_accuracy, val_accuracy)
                if val_accuracy > best_val:
                    best_val = val_accuracy
                    best_softmax = softmax
                else:
                    pass

        # Print out the results
        for learning_rate, reg in sorted(results):
            train_accuracy,val_accuracy = results[(learning_rate, reg)]
            print('learning rate %e and regularization %e, \n \
            the training accuracy is: %f and validation accuracy is: %f.\n' % (learning_rate, reg, train_accuracy, val_accuracy))

        y_test_predict_result = best_softmax.predict(X_test)
        y_test_predict = y_test_predict_result[0]
        test_accuracy = np.mean(oy_test == y_test_predict)
        accuracy.append(test_accuracy)
        totalAccuracy+=test_accuracy
        print('The test accuracy is: %f' % test_accuracy)
    print(accuracy)
    avgAccuracy = totalAccuracy / 5.0
    print('Average Accuracy: %f' % avgAccuracy)





