from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,RepeatVector, LSTM
from keras.models import Model
from keras import backend as K
import pickle as pic


def create_lstm_autoencoder(input_dim, timesteps, latent_dim):
	"""
	Creates an LSTM Autoencoder (VAE). Returns Autoencoder, Encoder, Generator. 
	
	"""
	inputs = Input(shape=(timesteps, input_dim))  # adapt this if using `channels_first` image data format
	encoded = LSTM(latent_dim)(inputs)

	decoded = RepeatVector(timesteps)(encoded)
	decoded = LSTM(input_dim, return_sequences=True)(decoded)

	sequence_autoencoder = Model(inputs, decoded)
	encoder = Model(inputs, encoded)
	
	autoencoder = Model(inputs, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	return autoencoder


