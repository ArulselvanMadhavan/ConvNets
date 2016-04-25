Readme.txt
1. Install python3. We suggest using Anaconda3 as it comes with almost all the packages that we use in this project.
2. Print the help options
python main.py -h
usage: use "main.py --help" for more information

details

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

 3. Download the data and save it in the data folder.
 	Note: Download the python version from here -> https://www.cs.toronto.edu/~kriz/cifar.html

 4. The raw data takes a lot of time to load. So, we stored the data in HDF5 format for faster loading.
 5. The data can be converted to HDF5 using the --loadCIFAR flag
 	Example: python main.py softmax -l
 	Note: You only need to do this once. Next, when you run the program, you can simply run it using python main.py softmax
 6. The first argument that you pass to the main.py is the algorithm that you want to run against the dataset.
 7. The flag '--features' indicates the feature selection algorithms that you want to use.
 	Example usage: If you want to use the HOG feature extraction->
 				   python main.py softmax -f 1 
 				   If you want to use the Histogram of colored bins feature extraction
 				   python main.py softmax -f 2
 				   If you want to use both
 				   python main.py softmax -f 1 2
 8. If you want to use ZCA,
 					python main.py softmax -f 1,2 -z
 9. If you want to run cnn, use the runme.ipynb notebook.
 10. To run that notebook, you must have the jupyter kernel installed.
 	 From the src directory start the kernel "jupyter notebook"
 11. Follow the steps in the notebook to test the CNN implementation.


 Note: We have tested our code on MacOSx machine. So, we'd prefer that you use a MacOSX machine to test our implementation.If you run into any issues while testing the code, please email us.