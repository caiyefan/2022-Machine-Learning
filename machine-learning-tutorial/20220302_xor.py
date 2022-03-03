import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))
y = np.array(([0], [1], [1], [0]))
Y = to_categorical(y)

model = keras.models.Sequential()
model.add(Input(2, name="input_layer"))
# model.add(Dense(4, activation="sigmoid"))
# model.add(Dense(4, activation="sigmoid", name="hidden_layer_1"))
model.add(Dense(2, activation="sigmoid", name="hidden_layer_2"))
model.add(Dense(2, activation="softmax", name="output_layer"))
model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["accuracy"])
model.fit(X, Y, epochs=1500)

# Make Test Dataset
test_X1, test_X2 = np.meshgrid(np.linspace(-1, 2, 30), np.linspace(-1, 2, 30))
test_X = np.stack((np.array(test_X1).flatten(), np.array(test_X2).flatten()), axis=1)
test_y = model.predict(test_X)
test_y = np.argmax(test_y, axis=1)
test_Xy = np.column_stack((test_X, test_y))


for i in range(len(test_y)):
    if test_y[i] == 0:
        plt.scatter(test_X[i][0], test_X[i][1], c='red')
    elif test_y[i] == 1:
        plt.scatter(test_X[i][0], test_X[i][1], c='green')
plt.xlim([-1, 2])
plt.ylim([-1, 2])
plt.show()

inp = model.input
outputs = [layer.output for layer in model.layers]
functors = [K.function([inp], [out]) for out in outputs]
layer_out = [func([test_X]) for func in functors]

for i in range(len(layer_out)):
    # if i==0:
    #     continue
    out = layer_out[i][0]
    for j in range(len(out)):
        if test_y[j] == 0:
            plt.scatter(out[j][0], out[j][1], c='red')
        elif test_y[j] == 1:
            plt.scatter(out[j][0], out[j][1], c='green')
    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.show()
