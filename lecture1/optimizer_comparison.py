import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create some synthetic data (a simple function to learn)
X = np.linspace(-2, 2, 1000).reshape(-1, 1)
y = X**2  # We'll try to learn y = x^2

# Define a simple model architecture
def create_model(optimizer):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Create models with different optimizers
optimizers = {
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.01),
    'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=0.01),
    'Adadelta': tf.keras.optimizers.Adadelta(learning_rate=1.0),  # Adadelta usually uses higher learning rate
}

# Train models and store history
histories = {}
for name, optimizer in optimizers.items():
    print(f"\nTraining with {name} optimizer...")
    model = create_model(optimizer)
    history = model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    histories[name] = history.history['loss']

# Plot training curves
plt.figure(figsize=(10, 6))
for name, loss in histories.items():
    plt.plot(loss, label=name)

plt.title('Optimizer Comparison')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.yscale('log')  # Using log scale to better see the differences
plt.legend()
plt.grid(True)
plt.show()

# Let's also print the final loss for each optimizer
print("\nFinal loss values:")
for name, loss in histories.items():
    print(f"{name}: {loss[-1]:.6f}")
