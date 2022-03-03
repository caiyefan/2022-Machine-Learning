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


scatter_x1 = test_X[:,0]
scatter_x2 = test_X[:,1]
cdict = {0:'red', 1:'green'}

fig, ax = plt.subplots()
for y in np.unique(test_y):
    ix = np.where(test_y == y)
    ax.scatter(scatter_x1[ix], scatter_x2[ix], c = cdict[y], label = y, s = 20)
ax.legend()
plt.show()

inp = model.input
outputs = [layer.output for layer in model.layers]
functors = [K.function([inp], [out]) for out in outputs]
layer_out = [func([test_X]) for func in functors]

for i in range(len(layer_out)):
    # if i==0:
    #     continue
    out = layer_out[i][0]

    scatter_x1 = out[:, 0]
    scatter_x2 = out[:, 1]

    fig, ax = plt.subplots()
    for y in np.unique(test_y):
        ix = np.where(test_y == y)
        ax.scatter(scatter_x1[ix], scatter_x2[ix], c=cdict[y], label=y, s=20)
    ax.legend()
    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.show()
