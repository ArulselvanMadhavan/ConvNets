"""
Acts as the entry point.
Main program that calls other programs
"""
import numpy as np
from LoadDataset import load_CIFAR_Dataset, getCIFAR_as_32Pixels_Image
import h5py
import argparse
from Features import Features

from ZCAWhitening import zca

"""
GLOBAL CONSTANTS
"""
TRAIN_DATASET_ID = 'X_train'
TRAIN_LABELS_ID = 'y_train'
TEST_DATASET_ID = 'X_test'
TEST_LABELS_ID = 'y_test'
PATH_TO_DATA_DIR = "../data/"
PATH_TO_TRAIN_FILE = PATH_TO_DATA_DIR + 'train.h5'
PATH_TO_TEST_FILE = PATH_TO_DATA_DIR + 'test.h5'
DATA_FOLDER_PATH = PATH_TO_DATA_DIR + 'cifar-10-batches-py'
ALGORITHM_ARGS = 'algo'  # Values are hardcoded in argsparser
KNN_ARGS = "knn"
SVM_ARGS = "svm"
ZCA_ARGS = "zca"
HOG_ARGS = "hog"
DEFAULT_CROSS_VALIDATION_FOLDS = 5


def getDataset(args):
    """
    Gets the dataset from the H5 files or from the disk based on the args
    :param args:
    :return: Train and Test data
    """
    if args.loadCIFAR:
        X_train, y_train, X_test, y_test = load_CIFAR_Dataset(DATA_FOLDER_PATH)

        with h5py.File(PATH_TO_TRAIN_FILE, 'w') as hf:
            hf.create_dataset(TRAIN_DATASET_ID, data=X_train)
            hf.create_dataset(TRAIN_LABELS_ID, data=y_train)

        with h5py.File(PATH_TO_TEST_FILE, 'w') as hf:
            hf.create_dataset(TEST_DATASET_ID, data=X_test)
            hf.create_dataset(TEST_LABELS_ID, data=y_test)

    else:
        print("Loading {} file".format(PATH_TO_TRAIN_FILE))
        with h5py.File(PATH_TO_TRAIN_FILE, 'r') as hf:
            print('List of arrays in this file: \n', hf.keys())
            data = hf.get(TRAIN_DATASET_ID)
            X_train = np.array(data)
            data = hf.get(TRAIN_LABELS_ID)
            y_train = np.array(data)

        print("Loading {} file".format(PATH_TO_TEST_FILE))
        with h5py.File(PATH_TO_TEST_FILE, 'r') as hf:
            print('List of arrays in this file: \n', hf.keys())
            data = hf.get(TEST_DATASET_ID)
            X_test = np.array(data)
            data = hf.get(TEST_LABELS_ID)
            y_test = np.array(data)
    print("{}\t{}\t{}\t{}\n".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    return X_train, y_train, X_test, y_test


def getFeatureFunctions(args):
    """
    Based on the user's choice pass the appropriate functions
    :param args:
    :return:
    """
    functionsArray = Features.getSupportedFunctions()
    featureFunctions = functionsArray[args.features]
    return Features(featureFunctions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='details',
                                     usage='use "%(prog)s --help" for more information',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-l", "--loadCIFAR", help="loads the data in ../data folder",
                        action="store_true")
    parser.add_argument(ALGORITHM_ARGS, help="Enter the algorithm to run(in lowercase)", choices=[KNN_ARGS, SVM_ARGS])
    parser.add_argument("-f", "--features",
                        help="Enter the feature selection Algorithm(s) Index of your choice\n"
                             "1.HOG\n"
                             "2.ZCA\n",
                        nargs='+',
                        type=int,
                        choices=[0, 1, 2])
    args = parser.parse_args()

    if args.algo == KNN_ARGS:
        X_train, y_train, X_test, y_test = getDataset(args)

        #TO-DO When KNN is implemented, move this into their preprocessing step
        #TO-DO Feature Extraction takes time, save them into h5 file and load them directly
        if args.features:
            X_train = getCIFAR_as_32Pixels_Image(X_train)
            X_test = getCIFAR_as_32Pixels_Image(X_test)
            ftsObj = getFeatureFunctions(args)
            # X_train = ftsObj.extract_features(X_train)
            X_test = ftsObj.extract_features(X_test)
        print(X_test.shape)
        print("KNN method yet to be implemented")
    elif args.algo == SVM_ARGS:
        X_train, y_train, X_test, y_test = getDataset(args)
        print("SVM yet to be implemented")
        # Add other algorithms like logistic regression here
