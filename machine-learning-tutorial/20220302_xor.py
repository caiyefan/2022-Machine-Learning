import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from keras import backend as K


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))
y = np.array(([0], [1], [1], [0]))
Y = to_categorical(y)

model = keras.models.Sequential()
model.add(Input(2, name="input_layer"))
# model.add(Dense(4, activation="sigmoid"))
model.add(Dense(4, activation="sigmoid", name="hidden_layer_1"))
model.add(Dense(2, activation="sigmoid", name="hidden_layer_2"))
model.add(Dense(2, activation="softmax", name="output_layer"))
model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["accuracy"])
model.fit(X, Y, epochs=1500)

inp = model.input
outputs = [layer.output for layer in model.layers]
functors = [K.function([inp], [out]) for out in outputs]
# functors = K.function([inp, K.learning_phase()], outputs)
# test = np.random.random(X.shape)[np.newaxis,...]
layer_out = [func([X]) for func in functors]
print(layer_out)
print(len(layer_out))
for i in range(len(layer_out)):
    if i==0:
        continue
    out = layer_out[i][0]
    plt.scatter(out[0][0], out[0][1], c='red')
    plt.scatter(out[1][0], out[1][1], c='green')
    plt.scatter(out[2][0], out[2][1], c='blue')
    plt.scatter(out[3][0], out[3][1], c='yellow')
    plt.show()

# for i in range(len(y)):
#     if y[i] == [0]:
#         plt.scatter(X[i][0], X[i][1], c='red')
#     elif y[i] == [1]:
#         plt.scatter(X[i][0], X[i][1], c='green')
# plt.show()
