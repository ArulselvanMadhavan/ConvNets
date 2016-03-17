import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
import matplotlib

class Features(object):
    """
    All Feature related functions go here
    """
    def __init__(self,featureFuncList):
        self.featureFuncList = featureFuncList

    @staticmethod
    def getSupportedFunctions():
        return np.array([Features.hogWrapper,Features.hsvWrapper,]) #Add ZCA here

    @staticmethod
    def hogWrapper(img):
        """
        Apply HOG on an image and return features of size (img_x_shape/8,img_y_shape/8)
        :param img:
        :return:
        """
        return hog(Features.rgb2gray(img),
                   orientations=9,
                   pixels_per_cell=(8,8),
                   cells_per_block= (1,1),
                   visualise=False,
                   normalise=False)

    @staticmethod
    def rgb2gray(img):
        return rgb2gray(img)

    @staticmethod
    def hsvWrapper(img):
        """
        A call to zca implementation
        :param img:
        :return:
        """
        return Features.hsv(img)

    @staticmethod
    def hsv(image,colorBins=10,xRangeMin=0,xRangeMax=255,isNormalized=True):
        ndim = image.ndim
        bins = np.linspace(0, 255, colorBins+1)
        hsv = matplotlib.colors.rgb_to_hsv(image/xRangeMax) * xRangeMax
        imageHistogram, bin_edges = np.histogram(hsv[:,:,0], bins=colorBins, density=isNormalized)
        imageHistogram = imageHistogram * np.diff(bin_edges)
        return imageHistogram

    def extract_features(self,imgs):
        num_images = imgs.shape[0]
        if num_images == 0:
            return np.array([])

        # Use the first image to determine feature dimensions
        feature_dims = []
        first_image_features = []
        for feature_fn in self.featureFuncList:
            feats = feature_fn(imgs[0].squeeze())
            feature_dims.append(feats.size)
            first_image_features.append(feats)

        total_feature_dim = sum(feature_dims)
        print("Total Features:{}".format(total_feature_dim))
        imgs_features = np.zeros((num_images, total_feature_dim))
        imgs_features[0] = np.hstack(first_image_features).T

        # Extract features for the rest of the images.
        for i in range(1, num_images):
            idx = 0
            for feature_fn, feature_dim in zip(self.featureFuncList, feature_dims):
                next_idx = idx + feature_dim
                imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
                idx = next_idx
            if i % 1000 == 0:
                print('Completed %d / %d images' % (i, num_images))
        return imgs_features
