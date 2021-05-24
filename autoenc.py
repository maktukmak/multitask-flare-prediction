import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.activations import relu

input_shape = (48, 36, 72, 1)
x = tf.random.normal(input_shape)


h = Conv2D(64, 3, (2,2), activation='relu', padding='same')(x)
h = Conv2D(64, 3, (2,2), activation='relu', padding='same')(h)
h = Flatten()(h)
h = Dense(16)(h)

h = Reshape((9,18,64))(Dense(9*18*64, activation='relu')(h))
h = Conv2DTranspose(64, 3, 2, activation='relu', padding='same')(h)
h = Conv2DTranspose(64, 3, 2, activation='relu', padding='same')(h)
y = Conv2DTranspose(1, 3, 1, activation='relu', padding='same')(h)

print(y.shape)
print(h.shape)