# custom L1 Distance Layer Module

import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance Layer from Jupyter notebook
class L1Dist(Layer):
    """ custom L1 layer to calculate L1 distance between two images by subtracting input embeddings from validation embeddings tp find the distance between them

    Args:
        Layer : base class
    """
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, input_embeddings, validation_embeddings):
        """return the L1 distance between two images i.e performing similarity calculation

        Args:
            input_embeddings : anchor image
            validation_embeddings : positive or negative image

        Returns:
            _type_: _description_
        """
        # calculate the distance between anchor or input image and positive or negative image
        distance = tf.math.abs(input_embeddings - validation_embeddings)
        return distance