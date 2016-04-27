
# coding: utf-8

# **ConvNets**
# 
# Some of our functions depend on modules written in C. You need to build them first to be able to import them into python. Run the following command from src/cnn/Layers directory
# ```python
# python setup_image.py build_ext --inplace
# ```
# Run the following cells in order.

# In[26]:

import numpy as np
from cnn.Worker import Worker
from cnn.GradientUpdates.StochasticGradientUpdate import StochasticGradientUpdate
from cnn.GradientUpdates.Adam import Adam
from cnn.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork


# In[ ]:

filters_list = [128]
dims,out_dims = ConvolutionalNeuralNetwork.generateConvLayerDimensions(filters_list,filter_size=3)
modl = ConvolutionalNeuralNetwork(dims,out_dims,weight_scale=0.05,filter_size=3,reg=0.05)
worker = Worker(modl,Adam(1e-3),
                train_size=49000,
                epochs_count=20,
                batch_size=50,
                lr_decay=0.95,
                debug=True,
                debug_every=1000)
worker.train()
print("Test Accuracy:{}".format(worker.test()))


# In[ ]:



