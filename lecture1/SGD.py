import tensorflow as tf

model = tf.keras.Sequential([...])

# choose an optimizer
optimizer = tf.keras.optimizers.SGD()

while True:

    prediction = model(x)

    with tf.GradientTape() as tape:
        #compute loss
        loss = compute_loss(y, prediction)

    #update the weights
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))