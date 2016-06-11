from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
from os.path import isfile
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.models import model_from_json
from keras.callbacks import TensorBoard

data = np.load("video.npy")
_, height, width = data.shape
data = data.reshape([-1, 1, height, width])
data = data.astype('float32') / 255.
print(data.shape)
print('[+] Building CNN')
input_img = Input(shape = (1, height, width))

x = Convolution2D(16, 3, 3, activation = 'relu', border_mode = 'same')(input_img)
x = MaxPooling2D((2, 2), border_mode = 'same')(x)
x = Convolution2D(8, 3, 3, activation = 'relu', border_mode = 'same')(x)
x = MaxPooling2D((2, 2), border_mode = 'same')(x)
x = Convolution2D(8, 3, 3, activation = 'relu', border_mode = 'same')(x)
encoded = MaxPooling2D((2, 2), border_mode = 'same')(x)

x = Convolution2D(8, 3, 3, activation = 'relu', border_mode = 'same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation = 'relu', border_mode = 'same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation = 'relu')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(1, 3, 3, activation = 'sigmoid', border_mode = 'same')(x)
decoded = ZeroPadding2D(padding = (2, 2), dim_ordering='th')(x)


model = Model(input_img, decoded)
model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
if isfile('compression_video.tflearn'):
	model.load_weights('compression_video.tflearn')
	print('[+] Model loaded')
else:
	print('[+] Training')
	model.fit(data, data,
                nb_epoch = 50,
                batch_size = 128,
                shuffle = True,
                validation_data = (data, data),
                callbacks = [TensorBoard(log_dir = '/tmp/autoencoder')])
	model.save_weights('compression_video.tflearn')
	print('[+] Model Saved')
print('[+] Compressing')
compress = model.predict(data)
np.save('video-compressed.npy', compress)