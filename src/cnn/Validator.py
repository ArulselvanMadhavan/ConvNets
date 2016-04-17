import numpy as np

class Validator(object):

    @staticmethod
    def get_accuracy(model, X, y, num_samples=None, fold_size=100):
        """
        Calculate the accuracy on the given input using Cross validation
        By sampling the given input in batches and averaging the overall accuracy
        :param model: A training model with a loss function
        :param X: Input Matrix
        :param y: Expected Results
        :param num_samples: Sample
        :param batch_size:
        :return:
        """
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        folds = N // fold_size
        if N % fold_size != 0:
            folds += 1
        y_pred = []
        for i in range(folds):
            start = i * fold_size
            end = (i + 1) * fold_size
            scores = model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc