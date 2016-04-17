import numpy as np


class Validator(object):
    @staticmethod
    def get_accuracy(model, X, y, sample_size=None, fold_size=100):
        """
        Calculate the accuracy on the given input using Cross validation
        By sampling the given input in batches and averaging the overall accuracy
        :param model: A training model with a loss function
        :param X: Input Matrix
        :param y: Expected Results
        :param num_samples: Sample
        :param fold_size: size of the fold
        :return:
        """
        N = X.shape[0]

        folds = N // fold_size
        if N % fold_size != 0:
            folds += 1
        y_pred = []
        if sample_size is not None and N > sample_size:
            N = sample_size
            X, y = Validator.get_samples(sample_size, N, X, y)
        for i in range(folds):
            start = i * fold_size
            end = (i + 1) * fold_size
            scores = model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc

    @staticmethod
    def get_samples(sample_size, N, X, y):
        """
        Generate random samples from input
        :param sample_size:
        :param N:
        :param X:
        :param y:
        :return:
        """
        mask = np.random.choice(N, sample_size)
        X = X[mask]
        y = y[mask]
        return X, y
