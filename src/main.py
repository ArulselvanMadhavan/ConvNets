"""
Acts as the entry point.
Main program that calls other programs
"""
import numpy as np
from LoadDataset import load_CIFAR_Dataset
import h5py
import argparse


from ZCAWhitening import zca,test_zca,construct_image,construct_ZCAimage

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
    print("{}\t{}\t{}\t{}\n".format(X_train.shape,y_train.shape,X_test.shape,y_test.shape))
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loadCIFAR", help="loads the data in ../data folder",
                        action="store_true")
    parser.add_argument(ALGORITHM_ARGS, help="enter the algorithm to run(in lowercase)", choices=[KNN_ARGS, SVM_ARGS])
    args = parser.parse_args()
    if args.algo == KNN_ARGS:
        X_train, y_train, X_test, y_test = getDataset(args)
        print(X_test)
        name1 = "OriginalTestImage.png"
        construct_image(X_test,y_test,name1)
        XZ_test = test_zca(X_test)
        print(XZ_test)
        name2 = "ZCATestImage.png"
        construct_ZCAimage(XZ_test,y_test,name2)
        print("KNN method yet to be implemented")
    elif args.algo == SVM_ARGS:
        X_train, y_train, X_test, y_test = getDataset(args)
        print("SVM yet to be implemented")
    #Add other algorithms like logistic regression here