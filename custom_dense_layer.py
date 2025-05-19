import tensorflow as tf
from tensorflow.keras.layers import Layer

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()
        # Initialize weights and bias
        self.W = self.add_weight(shape=(input_dim, output_dim))
        self.b = self.add_weight(shape=(1, output_dim))
    
    def call(self, inputs):
        # Implement the forward pass
        return tf.matmul(inputs, self.W) + self.b
