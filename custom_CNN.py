import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
import keras
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
# from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Model
from imblearn.over_sampling import RandomOverSampler
from keras.preprocessing import image

data = pd.read_csv('./icml_face_data.csv')
Y = data['emotion']
# Y = Y.reshape(Y.shape[0], 1)
print(Y.shape)
img_pixel = data[' pixels']
print(img_pixel.shape)
data.head()

oversampler = RandomOverSampler(sampling_strategy='auto')

X_over, Y_over = oversampler.fit_resample(img_pixel.values.reshape(-1,1), Y)

X_over_series = pd.Series(X_over.flatten())
X_over_series

def preprocess_data(pixel_data):
    images = []
    for p in range(len(pixel_data)):
        img = np.fromstring(pixel_data[p], dtype='int', sep = ' ')
        img.reshape(48,48,1)
        images.append(img)

    X = np.array(images)
    return X

# plt.imshow(X[0].reshape(48,48,1))
# plt.imshow(X[0,:,:,0])
# print(Y.shape[0])
# Y = Y.values.reshape(Y.shape[0], 1)
# print(Y.shape)

X = preprocess_data(X_over_series)
Y_ = Y_over
Y_ = Y_over.values.reshape(Y_.shape[0],1)
Y_.shape

x_train, x_test, y_train, y_test = train_test_split(X, Y_, test_size=0.1, random_state = 45)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
])




# def emotion_recognition(input_shape):

#   X_input = Input(input_shape)

#   X = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='valid')(X_input)
#   X = BatchNormalization(axis=3)(X)
#   X = Activation('relu')(X)


#   X = Conv2D(64, (3,3), strides=(1,1), padding = 'same')(X)
#   X = BatchNormalization(axis=3)(X)
#   X = Activation('relu')(X)

#   X = MaxPooling2D((2,2))(X)

#   X = Conv2D(64, (3,3), strides=(1,1), padding = 'valid')(X)
#   X = BatchNormalization(axis=3)(X)
#   X = Activation('relu')(X)

#   X = Conv2D(128, (3,3), strides=(1,1), padding = 'same')(X)
#   X = BatchNormalization(axis=3)(X)
#   X = Activation('relu')(X)


#   X = MaxPooling2D((2,2))(X)

#   X = Conv2D(128, (3,3), strides=(1,1), padding = 'valid')(X)
#   X = BatchNormalization(axis=3)(X)
#   X = Activation('relu')(X)



#   X = MaxPooling2D((2,2))(X)
#   X = Flatten()(X)
#   X = Dense(200, activation='relu')(X)
#   X = Dropout(0.6)(X)
#   X = Dense(7, activation = 'softmax')(X)

#   model = Model(inputs=X_input, outputs=X)

#   return model

# model = emotion_recognition((48,48,1))


model.summary()

adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

Y_train = to_categorical(y_train, num_classes=7)
print(Y_train.shape)
# print(x_train.shape)
X_train = x_train.reshape(56630,48,48,1)
X_test = x_test.reshape(6293, 48, 48, 1)
Y_test = to_categorical(y_test, num_classes=7)
history = model.fit(X_train, Y_train, epochs=30, validation_data=(X_test, Y_test))


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

test_score = model.evaluate(X_test, y_test, verbose = 0)
print("Test Score :-   ", test_score[0])
print("Test Accuracy :-   ", test_score[1])

model.save('./custom_CNN.h5')
