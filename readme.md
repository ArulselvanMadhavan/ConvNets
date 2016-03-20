1. Instructions to run the code goes here.
2. Track all softwares and their versions here. It is the responsibility of the new library/package adder to update this file.

<b>Dependencies</b>

1. NumPy -- pip install numpy
2. h5py -- pip install h5py


<b>Load Dataset</b>

1. python main.py -h -> Help command

2. python main.py -l knn  -> For the first time you run the code. This loads the dataset from your data folder and converts them into h5 file

3. python main.py knn -> Loads the h5 files directly 

<b>Run Cross Validation</b>
CrossValidation.py is a simple utility to generate the cross validation folds.

Here is a template for using it.
```python

    
    1. Create an object.
    
    crossValidObj = CrossValidation(numOfFolds, allData, allLabels)
    
    2. Generate Train and test
    
    foldsGen = crossValidObj.generateTrainAndTest()
    
    
    
    3. Iterate over the num of folds and access the train and test data
    
    
    for i in xrange(numOfFolds):
        next(foldsGen)
            crossValidObj = CrossValidation(numOfFolds, allData, allLabels)
    foldsGen = crossValidObj.generateTrainAndTest()
    for i in xrange(numOfFolds):
        next(foldsGen)
        X_train = crossValidObj.train
        y_train = crossValidObj.labels_train
        X_test = crossValidObj.test
        y_test = crossValidObj.labels_test
        //Call Whatever method you want.
        //Average the accuracy
```
