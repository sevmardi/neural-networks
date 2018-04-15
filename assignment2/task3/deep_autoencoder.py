from keras.layers import Input, Dense
from keras.models import Model
import pickle as pic
import time
import resource
from keras import regularizers
from keras.datasets import mnist
import numpy as np


start_time = time.time()

# import resource
# resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))

# this is the size of our encoded representations
# 32 floats -> comparasion of factor 24.5, assuming the input is 784  floats
encoding_dim = 32

# This is our placeholder
input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)


# "decoded" is the lossy reconstruction of the input
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# this model maps in input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# Load the data
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)
# print(x_test.shape)

#traing the model
autoencoder.fit(x_train, x_train, epochs=100, batch_size=256,
                shuffle=True, validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

pic.dump(decoded_imgs, open("pickles/deep_100.pickle", "wb"))
print('Success dump!')

# data_pic = "decoded_imgs_100runs.pickle"
# decoded_imgs = pic.load(open(data_pic, 'rb'))


# import matplotlib.pyplot as plt
# n = 10  # digits to display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display orginal
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)


# plt.savefig('plot_100_runs_regularization.png')
