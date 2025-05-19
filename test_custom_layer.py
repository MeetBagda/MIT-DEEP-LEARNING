import tensorflow as tf
from custom_dense_layer import MyDenseLayer
import numpy as np

# Create a simple model with our custom layer
model = tf.keras.Sequential([
    MyDenseLayer(input_dim=3, output_dim=2),
    tf.keras.layers.Activation('relu')
])

# Create some sample data
x = np.random.random((4, 3))  # 4 samples, 3 features each
print("Input shape:", x.shape)

# Run the model
output = model(x)
print("Output shape:", output.shape)
print("Output values:\n", output.numpy())
