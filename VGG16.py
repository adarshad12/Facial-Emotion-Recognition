import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
import keras
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from imblearn.over_sampling import RandomOverSampler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Model
from imblearn.over_sampling import RandomOverSampler
from keras.preprocessing import image
from keras.applications import VGG16

data = pd.read_csv('./icml_face_data.csv')
# print(df)
Y = data['emotion']
# Y = Y.reshape(Y.shape[0], 1)
print(Y.shape)
img_pixel = data[' pixels']
print(img_pixel.shape)
data.head()

oversampler = RandomOverSampler(sampling_strategy='auto')

X_over, Y_over = oversampler.fit_resample(img_pixel.values.reshape(-1,1), Y)

X_over_series = pd.Series(X_over.flatten())
# X_over_series

def preprocess_data(pixel_data):
    images = []
    for p in range(len(pixel_data)):
        img = np.fromstring(pixel_data[p], dtype='int', sep = ' ')
        rgb_img = np.repeat(img[..., np.newaxis], 3, -1)
        # print(rgb_img.shape)
        rgb_img.reshape(48,48,3)
        images.append(rgb_img)

    X = np.array(images)
    print(X.shape)
    return X

X = preprocess_data(X_over_series)
Y_ = Y_over
Y_ = Y_over.values.reshape(Y_.shape[0],1)
# Y_.shape

print(X.shape)
print(Y_.shape)

X = X.reshape(62923, 48, 48, 3)

X_train, X_test, y_train, y_test = train_test_split(X, Y_, test_size=0.2, random_state = 45)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.2, random_state=45)
# X_train = X.reshape(48,48,3)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)
print(y_validation.shape)
print(y_train.shape)
print(y_test.shape)

data_aug = ImageDataGenerator(width_shift_range=0.1, 
                              height_shift_range=0.1,
                              zoom_range=0.2,
                              shear_range=0.1,
                              rotation_range=10)


y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)
y_validation = to_categorical(y_validation, 7)

train_data = data_aug.flow(X_train, y_train, color_mode = "rgb",batch_size=10)
validation_data = data_aug.flow(X_validation, y_validation, color_mode = "rgb", batch_size=8)



vgg_model = VGG16(input_shape = (48,48,3), include_top = False, weights = 'imagenet')

for layer in vgg_model.layers[:-4]:
    layer.trainable = False

model = tf.keras.models.Sequential([
        vgg_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
])

model.summary()

adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Y_train = to_categorical(y_train, num_classes=7)
# print(Y_train.shape)
# print(x_train.shape)
# X_train = x_train.reshape(56630,48,48,1)
# X_test = x_test.reshape(6293, 48, 48, 1)
# Y_test = to_categorical(y_test, num_classes=7)


history = model.fit(train_data, epochs=30, validation_data=(X_validation, y_validation))

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

model.save('./predict_VGG.h5')
