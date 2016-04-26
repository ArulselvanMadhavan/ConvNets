
# coding: utf-8

# **ConvNets**
# 
# Some of our functions depend on modules written in C. You need to build them first to be able to import them into python. Run the following command from src/cnn/Layers directory
# ```python
# python setup_image.py build_ext --inplace
# ```
# Run the following cells in order.

# In[5]:

import numpy as np
from cnn.Worker import Worker
from cnn.GradientUpdates.StochasticGradientUpdate import StochasticGradientUpdate
from cnn.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork


# In[8]:

filters_list = [32]
dims,out_dims = ConvolutionalNeuralNetwork.generateConvLayerDimensions(filters_list)
modl = ConvolutionalNeuralNetwork(dims,out_dims,weight_scale=0.05,filter_size=3)
worker = Worker(modl,StochasticGradientUpdate(1e-3),
                train_size=100,
                epochs_count=20,
                batch_size=50,
                lr_decay=0.95,
                debug=True,
                debug_every=1)
worker.train()
worker.test()


# In[ ]:



