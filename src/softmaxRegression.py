
import struct
import numpy as np
import array
import time
from linear_classfier import *
from LoadDataset import *
import scipy.sparse
import scipy.optimize
from loss_grad_softmax import loss_grad_softmax_vectorized,loss_grad_softmax_naive

###########################################################################################
""" The Softmax Regression class """
def execute_softmax(X_train,y_train,X_test,y_test):
    num_training=49000
    num_val=1000
    num_test=10000
    X_train = getCIFAR_as_32Pixels_Image(X_train)
    X_test = getCIFAR_as_32Pixels_Image(X_test)
    mask = range(num_training, num_training + num_val)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1)) # [49000, 3072]
    X_val = np.reshape(X_val, (X_val.shape[0], -1)) # [1000, 3072]
    X_test = np.reshape(X_test, (X_test.shape[0], -1)) # [10000, 3072]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T


    W = np.random.randn(10, X_train.shape[0]) * 0.001
    tic = time.time()
    loss_naive, grad_naive = loss_grad_softmax_naive(W, X_train, y_train, 0.0001)
    toc = time.time()
    print('Naive loss: %f, and gradient: computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vec, grad_vect = loss_grad_softmax_vectorized(W, X_train, y_train, 0.0001)
    toc = time.time()
    print('Vectorized loss: %f, and gradient: computed in %fs' % (loss_vec, toc - tic))

    # Compare the gradient, because the gradient is a vector, we canuse the Frobenius norm to compare them
    # the Frobenius norm of two matrices is the square root of the squared sum of differences of all elements
    diff = np.linalg.norm(grad_naive - grad_vect, ord='fro')
    # Randomly choose some gradient to check
    idxs = np.random.choice(X_train.shape[0], 10, replace=False)
    print(idxs)
    print(grad_naive[0, idxs])
    print(grad_vect[0, idxs])
    print('Gradient difference between naive and vectorized version is: %f' % diff)
    del loss_naive, loss_vec, grad_naive

    softmax_sgd = Softmax()
    tic = time.time()
    losses_sgd = softmax_sgd.train(X_train, y_train, method='sgd', batch_size=200, learning_rate=1e-6,
                  reg = 1e5, num_iters=1000, verbose=True, vectorized=True)
    toc = time.time()
    print('Traning time for SGD with vectorized version is %f \n' % (toc - tic))

    y_train_pred_sgd = softmax_sgd.predict(X_train)[0]
    print('Training accuracy: %f' % (np.mean(y_train == y_train_pred_sgd)))
    y_val_pred_sgd = softmax_sgd.predict(X_val)[0]
    print('Validation accuracy: %f' % (np.mean(y_val == y_val_pred_sgd)))

    learning_rates = [1e-5, 1e-8]
    regularization_strengths = [10e2, 10e4]
    results = {}
    best_val = -1
    best_softmax = None
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
    test_accuracy = np.mean(y_test == y_test_predict)
    print('The test accuracy is: %f' % test_accuracy)






