"""
Acts as the entry point.
Main program that calls other programs
"""

import os

import numpy as np
from LoadDataset import load_CIFAR_Dataset, getCIFAR_as_32Pixels_Image
import h5py
import argparse
from Features import Features
from softmaxRegression import execute_softmax
from knn_Implement import *
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
SOFTMAX_ARGS = "softmax"
HSV_ARGS = "hsv"
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
    For more details look into features class
    :param args:
    :return:
    """
    functionsArray = Features.getSupportedFunctions()
    functionsList = sub1FromList(args.features)
    featureFunctions = functionsArray[functionsList]
    return Features(featureFunctions)

def sub1FromList(list):
    return [x - 1 for x in list]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='details',
                                     usage='use "%(prog)s --help" for more information',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-l", "--loadCIFAR", help="loads the data in ../data folder",
                        action="store_true")
    parser.add_argument(ALGORITHM_ARGS, help="Enter the algorithm to run(in lowercase)", choices=[KNN_ARGS, SVM_ARGS, SOFTMAX_ARGS, ZCA_ARGS])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--features",
                        help="Enter the feature selection Algorithm(s) Index of your choice\n"
                             "1.HOG\n"
                             "2.Histogram\n",
                        nargs='+',
                        type=int,
                        choices=[1, 2])
    group.add_argument("-z","--zca",help="ZCA Whitening",action="store_true")
    args = parser.parse_args()

    if args.algo == KNN_ARGS:
        #TO-DO When KNN is implemented, move this into their preprocessing step
        #TO-DO Feature Extraction takes time, save them into h5 file and load them directly
        X_train, y_train, X_test, y_test = getDataset(args)
        if args.zca:
            print("ZCA Pre Processing Started")
            X_train = zca(X_train)
            X_test = zca(X_test)
            print("ZCA Pre Processing Completed")
        elif args.features:
            #Call Feature Extraction techiniques
            X_train, y_train, X_test, y_test = getDataset(args)
            X_train = getCIFAR_as_32Pixels_Image(X_train)
            X_test = getCIFAR_as_32Pixels_Image(X_test)
            ftsObj = getFeatureFunctions(args)
            X_train = ftsObj.extract_features(X_train)
            X_test = ftsObj.extract_features(X_test)
        print(X_test.shape)
        print("Started with implementing KNN")
        executeKNN(X_train, y_train, X_test, y_test)
#        print("KNN method yet to be implemented")
    if args.algo == SOFTMAX_ARGS:
        X_train, y_train, X_test, y_test = getDataset(args)
        X_train = getCIFAR_as_32Pixels_Image(X_train)
        X_test = getCIFAR_as_32Pixels_Image(X_test)
        if args.zca:
            print("ZCA Pre Processing Started")
            X_train = zca(X_train)
            X_test = zca(X_test)
            X_train = getCIFAR_as_32Pixels_Image(X_train)
            X_test = getCIFAR_as_32Pixels_Image(X_test)
            print("ZCA Pre Processing Completed")
            execute_softmax(X_train,y_train,X_test,y_test)
        elif args.features:
            #Call Feature Extraction techiniques
            X_train = getCIFAR_as_32Pixels_Image(X_train)
            X_test = getCIFAR_as_32Pixels_Image(X_test)
            ftsObj = getFeatureFunctions(args)
            X_train = ftsObj.extract_features(X_train)
            X_test = ftsObj.extract_features(X_test)
            execute_softmax(X_train,y_train,X_test,y_test)
        else:
            execute_softmax(X_train,y_train,X_test,y_test)
    if args.algo == ZCA_ARGS:
        print("This is just and experiment to see that the code works")
        X_train, y_train, X_test, y_test = getDataset(args)
        #construct_image(X_test,y_test,"original.png")
        XZ_test=zca(X_test)
        #construct_ZCAimage(XZ_test,y_test,"zca.png")
    elif args.algo == SVM_ARGS:
        X_train, y_train, X_test, y_test = getDataset(args)
        print("SVM yet to be implemented")
        # Add other algorithms like logistic regression here