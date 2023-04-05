from matplotlib import pyplot
from keras.datasets import cifar10
import tensorflow as tf

(trainX,trainY), (testX, testY) = cifar10.load_data()

trainY = tf.keras.utils.to_categorical(trainY).astype(int)
testY = tf.keras.utils.to_categorical(testY).astype(int)


print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))

print(trainY[0][0])

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(trainX[i])

pyplot.show()


def load_dataset():
    (trainX,trainY), (testX, testY) = cifar10.load_data()

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# train_norm = trainX.astype('float32')/255.0
# test_norm = testX.astype('float32')/255.0


def prep_pixels(train, test):
    train_norm = trainX.astype('float32')/255.0
    test_norm = testX.astype('float32')/255.0
    return train_norm, test_norm

def define_model():
    model = Sequential()
    return model


seqModel = tf.keras.Sequential()
seqModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss = tf.keras.losses.CategoricalCrossentropy())

results = seqModel.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=0)
_, acc = model.evaluate(testX, testY, verbose=0)

def summarize_diagnostics(results):
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(results.history['loss'], color='blue', label='train')
    pyplot.plot(results.history['val_loss'], color='orange', label='test')
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(results.history['accuracy'], color='blue', label='test')
    pyplot.plot(results.history['val_accuracy'], color='orange', label='test')


summarize_diagnostics(results)
