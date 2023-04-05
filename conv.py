from matplotlib import pyplot
from keras.datasets import cifar10

(trainX,trainy), (testXm, testy) = cifar10.load_data()

print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

import tensorflow as tf
