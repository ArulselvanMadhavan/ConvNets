import numpy as np


class CrossValidation(object):
    def __init__(self, numOfFolds, dataset, labels):
        """
        Constructor
        :param numOfFolds:
        :param dataset:
        :return:
        """
        self.numOfFolds = numOfFolds
        self.dataset = dataset
        self.labels = labels
        self.train = None
        self.test = None
        self.labels_train = None
        self.labels_test = None

    def generateTrainAndTest(self):
        """
        Generate train and test data and then yield
        :return:
        """
        partitions = np.array_split(self.dataset, self.numOfFolds)
        labels_partitions = np.array_split(self.labels, self.numOfFolds)
        for fold in range(self.numOfFolds):
            self.test = partitions[fold]
            self.labels_test = labels_partitions[fold]

            fold_left = partitions[:fold]
            fold_right = partitions[fold + 1:]

            labels_fold_left = labels_partitions[:fold]
            labels_fold_right = labels_partitions[fold + 1:]

            if fold_left.__len__() == 0:
                self.train = np.concatenate(fold_right)
                self.labels_train = np.concatenate(labels_fold_right)
            elif fold_right.__len__() == 0:
                self.train = np.concatenate(fold_left)
                self.labels_train = np.concatenate(labels_fold_left)
            else:
                self.train = np.concatenate((np.concatenate(fold_left), np.concatenate(fold_right)))
                self.labels_train = np.concatenate(
                        (np.concatenate(labels_fold_left), np.concatenate(labels_fold_right)))
            yield
