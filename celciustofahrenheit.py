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
import  matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

#predict the fahranheit given 100 degrees celsius
print(model.predict([100.0]))

#layers weights
print("These are the layer variables: {}".format(10.get_weights()))

# for fun
#reformat
10 = tf.keras.layers.Dense(units=4, input_shape=[1])
11 = tf.keras.layers.Dense(units=4)
12 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([10, 11, 12])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celcius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training")
print(model.predict([100.0]))
print("Model predicts that 100 degrees  Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))
print("These are the 10 variables: {}".format(10.get_weights()))
print("These are the 11 variables: {}".format(11.get_weights()))
print("These are the 12 variables: {}".format(12.get_weights()))