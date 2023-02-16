import tensorflow as tf
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#training data

celcius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i,c in enumerate(celcius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

#creat the model
10 = tf.keras.layers.Dense(units=1, input_shape=[1])

#assemble layers into the model
model = tf.keras.Sequential([10])
#model = tf.keras.Sequential([
#    tf
#])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

#Training the model
history = model.fit(celcius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

#Diplay training stats