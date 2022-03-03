import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from keras.datasets import mnist

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load Dataset
(train_x, train_y), (test_x, test_y_) = mnist.load_data()

# Dataset Pre-processing
train_x = train_x.reshape(train_x.shape[0], 784)
test_x = test_x.reshape(test_x.shape[0], 784)
train_y = to_categorical(train_y)
test_y = to_categorical(test_y_)
train_x = train_x / 255
test_x = test_x / 255

# model = keras.models.Sequential()
# model.add(Input(784, name="input_layer"))
# model.add(Dense(128, activation="sigmoid", name="hidden_layer_1"))
# model.add(Dense(64, activation="sigmoid", name="hidden_layer_2"))
# model.add(Dense(3, activation="sigmoid", name="hidden_layer_3"))
# model.add(Dense(10, activation="softmax", name="output_layer"))
# model.compile(loss="categorical_crossentropy",
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=["accuracy"])
# model.fit(train_x, train_y, batch_size=128, epochs=50)
# model.save("saved_model_mnist_3dim_20220303")

# model = keras.models.load_model("saved_model_mnist_3dim_20220303")
model = keras.models.load_model("/home/caiye/workspace/2022-Machine-Learning/machine-learning-tutorial/saved_model_mnist_3dim_20220303")

score = model.evaluate(test_x, test_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

inp = model.input
outputs = [layer.output for layer in model.layers]
functors = [K.function([inp], [out]) for out in outputs]
layer_out = [func([test_x]) for func in functors]
out = layer_out[2][0]

scatter_x1 = out[:,0]
scatter_x2 = out[:,1]
scatter_x3 = out[:,2]
cdict = {0:'blue', 1:'orange', 2:'green', 3:'red', 4:'purple',
         5:'brown', 6:'pink', 7:'gray', 8:'olive', 9:'cyan'}
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for y in np.unique(test_y_):
    ix = np.where(test_y_ == y)
    ax.scatter(scatter_x1[ix], scatter_x2[ix], scatter_x3[ix], c=cdict[y], label=y+1)
# ax.legend(loc='upper right')
plt.show()
