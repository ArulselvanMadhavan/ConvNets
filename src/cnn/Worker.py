import numpy as np
from LoadDataset import getCIFAR_as_32Pixels_Image
from main import getDataset


class Worker(object):
    def __init__(self, train_size=49000, test_size=10000, val_size=1000):
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = Worker.normalized_data(
                train_size, test_size, val_size)

    @staticmethod
    def normalized_data(self, train_size, test_size, val_size):
        """
        Normalize the input data
        Separate them into train, test and validation dataset
        :param train_size:
        :param test_size:
        :param val_size:
        :return:
        """
        X_train, Y_train, X_test, Y_test = getDataset(False)

        X_train = getCIFAR_as_32Pixels_Image(X_train)
        X_test = getCIFAR_as_32Pixels_Image(X_test)

        mask = range(train_size, train_size + val_size)
        X_val = X_train[mask]
        y_val = Y_train[mask]
        mask = range(train_size)
        X_train = X_train[mask]
        y_train = Y_train[mask]
        mask = range(test_size)
        X_test = X_test[mask]
        y_test = Y_test[mask]

        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

        X_train = X_train.transpose(0, 3, 1, 2).copy()
        X_val = X_val.transpose(0, 3, 1, 2).copy()
        X_test = X_test.transpose(0, 3, 1, 2).copy()

        return X_train, y_train, X_val, y_val, X_test, y_test
