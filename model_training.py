import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

X = np.load('D://deep learning//Mini Project//x_data.npy')
y = np.load('D://deep learning//Mini Project//y_data.npy')

classes = ['cheesecake','chicken_wings','donuts','french_fries','fried_rice',
           'hamburger','hot_dog','ice_cream','nachos','omelette','onion_rings',
           'pizza','salad','samosa','sushi']

def plot_images(X,y,index):
    plt.figure(figsize=(5,2))
    plt.imshow(X[index])
    plt.xlabel(classes[np.argmax(y[index])])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=0.2)

base = MobileNetV2(input_shape=(256,256,3),include_top=False,weights='imagenet')

model = keras.Sequential([
        base,
        GlobalAveragePooling2D(),
        layers.Dense(150, activation='relu'),
        layers.Dense(15, activation='softmax')
    ])

opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt,loss = 'categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1)
