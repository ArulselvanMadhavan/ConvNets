#Instructions to run the code
1. Install python3. We suggest using Anaconda3 as it comes with almost all the packages that we use in this project
2. Print the help options
	```python
	python main.py --help
	
	usage: use "main.py --help" for more information details
	
	positional arguments:
	  {knn,svm,softmax,zca,cnn}
	                        Enter the algorithm to run(in lowercase)
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -l, --loadCIFAR       loads the data in ../data folder
	  -f {1,2} [{1,2} ...], --features {1,2} [{1,2} ...]
	                        Enter the feature selection Algorithm(s) Index of your choice
	                        1.HOG
	                        2.Histogram
	  -z, --zca             ZCA Whitening
	```
	
3. **You may skip this step if you have the link to HDF5 format files.If you don't have the link to HDF5 files, email us for the link and proceed with step 3**
  * Download the data and save it in the data folder
  * Download the python version from here -> https://www.cs.toronto.edu/~kriz/cifar.html
  * The dataset can be parsed and loaded with python2. This is a restriction imposed by the dataset.
  * Install python2
  * Install numpy 
	```python
	pip install numpy
	```
  * Install h5py 
	```python
	pip install h5py
	```
  * Run the following command
	```python
	python main.py knn -l
	```
  * After you run the command, you should see train.h5,test.h5 files created in the data directory. Only for the first time,you have to use the "-l" flag. Once you have the 'h5' files created, you can discard the "-l" flag. 
  * VERY IMPORTANT: We use python2 only to parse the dataset. The rest of the project uses python3.
  * The raw data takes a lot of time to load. So, we stored the data in HDF5 format for faster loading
4. The first argument that you pass to the main.py is the algorithm that you want to run against the dataset.
	```python
	python main.py softmax
	
	python main.py knn
	
	python main.py svm
	```
5. The flag '--features' indicates the feature selection algorithms that you want to use.
  * To use the Histogram of Oriented Gradients
	```python
 	python main.py softmax -f 1 
 	```
  * To use Histogram of Colored Bins
  	```python
  	python main.py softmax -f 2
  	```
  * To use both feature extraction techniques
  	```python
 	python main.py softmax -f 1 2
 	```
  * To use ZCA,
 	```python
 	python main.py softmax -z
 	```
  * To use ZCA along with any feature extraction technique
  	```python
	python main.py softmax -z -f 1 2
	```
7. If you want to run cnn, you can either use the runme.ipynb notebook or the runme.py file.
  * Follow the instructions in the file to run the cnn code.
  * If you want to use the notebook, you must have the jupyter kernel installed.
8. The file by default has the best configurations that worked for us. If you want to change the configurations, you can do that by passing your configurations as arguments to the constructor.

 Note: We have tested our code on MacOSx machine. So, we'd prefer that you use a MacOSX machine to test our implementation.If you run into any issues while testing the code, please email us.
