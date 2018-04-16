from __future__ import print_function

import numpy as np

from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# For reproducibility
np.random.seed(1000)
save_dir = os.path.join(os.getcwd(), 'saved_models')	
model_name = 'keras_cifar10_trained_model.h5'

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                 optimizer=Adam(lr=0.0001, decay=1e-6),
                 metrics=['accuracy'])

    # train the model
    model.fit(X_train / 255.0, to_categorical(Y_train),
              batch_size=128,
              shuffle=True,
              epochs=250,
              validation_data=(X_test / 255.0, to_categorical(Y_test)),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

	

    if not os.path.isdir(save_dir):
    	os.makedirs(save_dir)
    
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


    scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

    decoded_imgs = model.predict(X_test)


    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])


    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
    	ax = plt.subplot(2, n, i + 1)
    	plt.imshow(x_test[i].reshape(28, 28))
    	plt.gray()
    	ax.get_xaxis().set_visible(False)
    	ax.get_yaxis().set_visible(False)

    	ax = plt.subplot(2, n, i + n)
    	plt.imshow(decoded_imgs[i].reshape(28, 28))
    	plt.gray()
    	ax.get_xaxis().set_visible(False)
    	ax.get_yaxis().set_visible(False)

    plt.savefig('plots/cifar.png')

