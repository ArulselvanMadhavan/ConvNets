import numpy as np
from LoadDataset import getCIFAR_as_32Pixels_Image
from main import getDataset
from cnn.GradientUpdates.BaseGradientUpdate import BaseGradientUpdate
import copy
from cnn.Validator import Validator
from cnn.BaseNeuralNetwork import BaseNeuralNetwork


class Worker(object):
    def __init__(self, conv_model, update, train_size=49000,
                 test_size=10000, val_size=1000, batch_size=100,
                 epochs_count=10, lr_decay=1.0, debug=True, debug_every=10):
        """
        :param conv_model: A Neural Network model
        :param update: A Gradient Update function
        :param train_size: Training Size
        :param test_size: Test size
        :param val_size: Validation size
        :param batch_size: Batch size
        :param epochs_count: Number of epochs
        :param lr_decay: Learning rate decay
        :param debug: Debug flag
        :param debug_every: How often you want to print the training loss
        :return: A worker object
        """
        self.model = conv_model
        assert isinstance(self.model, BaseNeuralNetwork)
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = Worker.normalized_data(
                train_size, test_size, val_size)
        assert isinstance(update, BaseGradientUpdate)
        self.batch_size = batch_size
        self.epochs_count = epochs_count
        self.epoch = 0
        self.best_val_acc = 0
        self.lr_decay = lr_decay
        self.debug = debug
        self.W_configs = []  # Stores the configurations specific to Gradient update functions on W
        self.b_configs = []  # Stores the configurations specific to Gradient update functions on b
        self.loss = 0.0
        self.debug_every = debug_every
        for i in range(len(self.model.layer_objs)):
            self.W_configs.append(copy.deepcopy(update))
            self.b_configs.append(copy.deepcopy(update))

    def _send_pulse(self):
        """
        Send a signal and do a gradient update
        Use Mini batch gradient descent
        """
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.Y_train[batch_mask]

        self.loss = self.model.loss(X_batch, y_batch)

        # Update weights and bias
        for layer_id in range(len(self.model.layer_objs)):
            layer_obj = self.model.layer_objs[layer_id]
            next_w = self.W_configs[layer_id].update(layer_obj.W, layer_obj.dw)
            layer_obj.W = next_w
            next_b = self.b_configs[layer_id].update(layer_obj.b, layer_obj.db)
            layer_obj.b = next_b

    def train(self):
        iterations_count, iterations_per_epoch = Worker.get_iterations_count(self.X_train.shape[0], self.batch_size,
                                                                             self.epochs_count)
        for t in range(iterations_count):
            self._send_pulse()

            # Maybe print training loss
            if self.debug and t % self.debug_every == 0:
                print('loss: %f after (Iteration %d / %d)' % (
                    self.loss,t + 1, iterations_count,))

            #Update learning rate after an epoch
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in range(len(self.model.layer_objs)):
                    self.W_configs[k].decay(self.lr_decay)
                    self.b_configs[k].decay(self.lr_decay)

            #Print accuracy after each epoch
            if epoch_end:
                train_acc = Validator.get_accuracy(self.model, self.X_train, self.Y_train,
                                                   sample_size=1000)
                val_acc = Validator.get_accuracy(self.model, self.X_val, self.Y_val)

                if self.debug:
                    print('(Epoch %d / %d) train_acc: %f; val_acc: %f' % (
                        self.epoch, self.epochs_count, train_acc, val_acc))

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc

                    self.best_Ws = []
                    self.best_bs = []
                    for layer_obj in self.model.layer_objs:
                        self.best_Ws.append(layer_obj.W)
                        self.best_bs.append(layer_obj.b)

        for layer_id in range(len(self.model.layer_objs)):
            self.model.layer_objs[layer_id].W = self.best_Ws[layer_id]
            self.model.layer_objs[layer_id].b = self.best_bs[layer_id]

    def test(self):
        """
        Test Data set
        :param X:
        :param y:
        :return:
        """
        return Validator.get_accuracy_with_confusion_matrix(self.model, self.X_test, self.Y_test)

    @staticmethod
    def normalized_data(train_size, test_size, val_size):
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

        return (X_train, y_train, X_val, y_val, X_test, y_test)

    @staticmethod
    def get_iterations_count(train_size, batch_size, epochs_count):
        iterations_per_epoch = max(train_size // batch_size, 1)
        iterations_count = epochs_count * iterations_per_epoch
        return iterations_count, iterations_per_epoch
