"""Importing libraries"""
import tensorflow as tf  # Will be used as a backend
import numpy as np
import pandas as pd
from PIL import Image  # Helps to open images
import os  # Helps to use operating system functionalities
from sklearn.model_selection import train_test_split  # Importing sklearn
from sklearn.metrics import accuracy_score  # Importing accuracy
from keras.utils import to_categorical  # Importing keras - Front
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
import pickle

"""Storing data and labels in the list"""
data = []  # Empty list of data
labels = []  # Empty list of labels
# classes = 43

"""Locating working directory and printing it out"""
file_path = os.getcwd()
# print(file_path)
# C:\Users\SMXaS\PycharmProjects\signs_recognition

"""
Looping through 43 classes
Opening data - train file
Declaring 'images' as faster way to open train file
---------------------------------------------------
Looping through settings of images
Opening images
Resizing the pictures
Modifying the list with append for data and labels
Raising the exception if there was any errors
"""
for pictures in range(43):
    data_path = os.path.join(file_path, 'Data/Train', str(pictures))
    images = os.listdir(data_path)
    for setting in images:
        try:
            image = Image.open(data_path + '\\' + setting)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(pictures)
        except Exception as error:
            print(error)

"""Making data and labels list to be numpy arrays"""
data = np.array(data)
labels = np.array(labels)

"""
Saving our model after training with Numpy
It can be used for future purposes
There is no need to go through all steps from beginning, use this
"""
np.save('./Data/Future_Data/data', data)
np.save('./Data/Future_Data/label', labels)

"""Loading Numpy Future Data"""
data = np.load('./Data/Future_Data/data.npy')
labels = np.load('./Data/Future_Data/labels.npy')

"""Printing out the shape of data and labels"""
# print(data.shape, labels.shape)

"""Preparing testing, training and setting up the parameters"""
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

"""Converting classes to encode integer data"""
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

"""
Preparing the model
Model has parameters from KERAS API
Activation layers, filters, size, shape, rate which can be changed
"""
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

"""Compiling the model"""
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""Setting up the epochs and history of each compile"""
epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

"""
Importing CSV
Setting up csv reading
predicting what we need
"""
data_csv = pd.read_csv("./Data/LabelsCSV.csv", sep=',')
data_csv = data_csv[["class_id", "name"]]
predict = ["name"]

"""Reading csv file..."""
def test_image(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["class_id"].values
    imgs = y_test["name"].values
    data = []
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30, 30))
        data.append(np.array(image))
    X_test = np.array(data)
    return X_test, label


X_test, label = test_image('Test.csv')
Y_predict = model.predict_classes(X_test)

print(accuracy_score(label, Y_predict))

model.save('./Data/Future_Data/TSR.h5')

model = load_model('./Data/Future_Data/TSR.h5')


def test_img(img):
    data = []
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))
    X_test = np.array(data)
    Y_predict = model.predict_classes(X_test)
    return image, Y_predict


plot, prediction = test_img(r'C:\Users\SMXaS\PycharmProjects\signs_recognition\Data\Test\00000.png')
s = [str(i) for i in prediction]
a = int("".join(s))
print("Predicted: ", data_csv[a])
plt.imshow(plot)
plt.show()
