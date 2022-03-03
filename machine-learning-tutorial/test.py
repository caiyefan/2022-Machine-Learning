import matplotlib.pyplot as plt
import numpy as np


# Make Test Dataset
# test_x = np.array([np.linspace(-1, 2, 30), np.linspace(-1, 2, 30)])
test_x1 = np.linspace(-1, 2, 30)
test_x2 = np.linspace(-1, 2, 30)
test_X1, test_X2 = np.meshgrid(test_x1, test_x2)

test_X1 = np.array(test_X1).flatten()
test_X2 = np.array(test_X2).flatten()
test_X = np.stack((test_X1, test_X2), axis=1)
print(test_X[0])

