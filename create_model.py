import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

# All available labels
labels = []

# Images
x = []
# Labels for images
y = []

image_paths = []

def load_paths():
    for root, dirs, files in os.walk(".", topdown=False): 
        for name in files:
            path = os.path.join(root, name)
            if path.endswith("png"):
                image_paths.append(path)
    print(f'Loaded {len(image_paths)} paths to images.')

def load_images(image_paths):
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (250, 250))
        x.append(img)
        label = path.split("/")[2]
        if label not in labels:
            labels.append(label)
        y.append(labels.index(label))

    X = np.array(x, dtype="uint8")
    X = X.reshape(len(image_paths), 250, 250, 1)
    Y = np.array(y)
    print(f'Loaded {len(X)} images.')
    print(f'Loaded {len(Y)} labels')
    print(labels)
    return X, Y

# Loading paths to images
load_paths()
# Loading all images
X, Y = load_images(image_paths)

# Splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Creating model
model = Sequential()
model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(250, 250, 1))) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=5, batch_size=50, verbose=1, validation_data=(X_test, y_test))
model.save('models/gesture_rec_model2.h5')

predictions = model.predict(X_test)
print(predictions)
y_pred = np.argmax(predictions, axis=1)

# Confusion matrix
conf = confusion_matrix(y_test, y_pred)
print(conf)