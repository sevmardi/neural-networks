from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import pickle as pic
from keras.callbacks import TensorBoard
import os 


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

save_dir = os.path.join(os.getcwd(), 'saved_models')	
model_name = '50_convolutional_autoencoder.h5'




input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format



autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


decoded_imgs = autoencoder.predict(x_test)

# pic.dump(decoded_imgs, open("pickles/convolutional_autoencoder_50.pickle", "wb"))
# data_pic = "pickles/convolutional_autoencoder_50.pickle"
# decoded_imgs = pic.load(open(data_pic, 'rb'))

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
autoencoder.save(model_path)
print('Saved trained model at %s ' % model_path)

# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
# 	# display original
# 	ax = plt.subplot(2, n, i + 1)
# 	plt.imshow(x_test[i].reshape(28, 28))
# 	plt.gray()
# 	ax.get_xaxis().set_visible(False)
# 	ax.get_yaxis().set_visible(False)

# 	# display reconstruction
# 	ax = plt.subplot(2, n, i + n)
# 	plt.imshow(decoded_imgs[i].reshape(28, 28))
# 	plt.gray()
# 	ax.get_xaxis().set_visible(False)
# 	ax.get_yaxis().set_visible(False)

# plt.savefig('plots/50_convolutional_autoencoder.png')