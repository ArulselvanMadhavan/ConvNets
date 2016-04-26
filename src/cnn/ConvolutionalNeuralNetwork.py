import numpy as np
from cnn.Layers.ConvLayer import ConvLayer
from cnn.Layers.LinearNet import LinearNet
from cnn.Layers.Linear_Rect import Linear_Rect
from cnn.Layers.Conv_Rect_Pool import Conv_Rect_Pool
from cnn.LossFunctions.Softmax_Loss import Softmax_Loss
from cnn.BaseNeuralNetwork import BaseNeuralNetwork


class ConvolutionalNeuralNetwork(BaseNeuralNetwork):
    _layer_map = {0: "Conv_Rect_Pool", 3: "Linear_Rect", 4: "LinearNet"}

    def __init__(self, layer_weight_dims, layer_output_dims, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 layer_config=[1, 0, 0, 1, 1]):
        """
        Layer Id
        0 - Conv_Rect_Pool
        1 - Conv_Rect
        2 - Conv
        3 - Linear_Rect
        4 - Linear
        Input list = [2,0,0,1,1] - 2 Conv_Rect_Pool layers, 1 Linear_Rect and 1 Linear
        """

        self.layer_objs = []
        self.layer_output_dims = layer_output_dims
        self.layer_weight_dims = layer_weight_dims
        assert len(layer_output_dims) == len(layer_weight_dims)
        self.num_conv_layers = len(layer_output_dims)

        # Construct the Conv layers
        for layer_weight_dim in self.layer_weight_dims:
            w, b = ConvLayer.generateWeightsAndBias(layer_weight_dim, weight_scale)
            conv_obj = Conv_Rect_Pool(w, b, pad=(filter_size - 1) // 2)
            self.layer_objs.append(conv_obj)

        # Construct the FC layer
        w2, b2 = LinearNet.generateWeightsAndBias(np.prod(self.layer_output_dims[-1]), hidden_dim,
                                                  weight_scale)
        lin_obj = Linear_Rect(w2, b2)

        # Construct the output layer
        w3, b3 = LinearNet.generateWeightsAndBias(hidden_dim, num_classes, weight_scale)
        lin_obj_2 = LinearNet(w3, b3)

        self.layer_objs.append(lin_obj)
        self.layer_objs.append(lin_obj_2)

        self.reg = reg

    def loss(self, X, y=None):
        N = X.shape[0]  # TODO-Figure out a way to infer N
        out = X
        for conv_layer in range(self.num_conv_layers):
            out = self.layer_objs[conv_layer].forward(out)
        out = np.reshape(out, ((N,) + self.layer_output_dims[-1]))
        out = self.layer_objs[-2].forward(out)
        scores = self.layer_objs[-1].forward(out)

        if y is None:
            return scores

        data_loss, dscores = Softmax_Loss.compute_loss_and_gradient(scores, y)
        reg_loss = 0
        for layer_obj in self.layer_objs:
            reg_loss += 0.5 * self.reg * np.sum(layer_obj.W ** 2)

        loss = data_loss + reg_loss

        out = self.layer_objs[-1].backward(dscores)
        self.layer_objs[-1].dw += (self.reg * self.layer_objs[-1].W)

        out = self.layer_objs[-2].backward(out)
        self.layer_objs[-2].dw += (self.reg * self.layer_objs[-2].W)

        out = np.reshape(out, ((N,) + self.layer_output_dims[-1]))

        for layer_id in range(self.num_conv_layers - 1, -1, -1):
            out = self.layer_objs[layer_id].backward(out)
            self.layer_objs[layer_id].dw += (self.reg * self.layer_objs[layer_id].dw)

        return loss

    def printAllLayerDimensions(self):
        for layer_obj in self.layer_objs:
            layer_obj.printDimensions()

    @staticmethod
    def generateConvLayerDimensions(num_filters_list, filter_size=3, image_dim=(3, 32, 32), pool_height=2, pool_width=2,
                                    pool_stride=2):
        """
        This code just assumes that conv layers always do a pad that maintains the original image size after
        convolution.
        :param num_filters_list:
        :param filter_size:
        :param image_dim:
        :param pool_height:
        :param pool_width:
        :param pool_stride:
        :return:
        """
        layers_count = len(num_filters_list)
        dims_list = []

        # Repeating the first element as a dirty hack
        # get the for loop working for all layers. Layer 1 is different as it looks at the input dimensions
        prev_layer_dim = (image_dim[0],) + image_dim
        layer_output_dim = []
        #This code assumes that the stride will always be 1
        stride = 1
        for layer in range(len(num_filters_list)):
            N = num_filters_list[layer]
            C = prev_layer_dim[0]
            pad = (filter_size - 1) // 2
            H_out = 1 + (prev_layer_dim[2] + 2 * pad - filter_size) //stride
            W_out = 1 + (prev_layer_dim[3] + 2 * pad - filter_size) // stride
            H = ConvolutionalNeuralNetwork.downsample(H_out, pool_height, pool_stride)
            W = ConvolutionalNeuralNetwork.downsample(W_out, pool_width, pool_stride)
            dims_list.append((N, C, filter_size, filter_size))
            prev_layer_dim = (N, C, H, W)
            layer_output_dim.append((N, H, W))
        return dims_list, layer_output_dim

    @staticmethod
    def downsample(dim, downsize, stride):
        return (dim - downsize) // stride + 1
