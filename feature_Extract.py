# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 10:12:34 2021

@author: Aravind
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
import cv2
import pickle

IMG_SIZE = 256
directory = 'D:\deep learning\Mini Project\images'
class_dir = [x[0] for x in os.walk('D:\deep learning\Mini Project\images')]
categories = []

for index,url in enumerate(class_dir):
    if index == 0:
        continue
    categories.append(url.split('\\')[-1])

training_data = []

def create_training_data():
    for category in categories:
        path = os.path.join(directory, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception:
                pass
create_training_data()

random.shuffle(training_data)
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
    
x_np = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_np = np.array(y).reshape(-1)

y_np = tf.keras.utils.to_categorical(y_np, len(categories))

np.save('D://deep learning//Mini Project//x_data.npy', x_np)
np.save('D://deep learning//Mini Project//y_data.npy', y_np)

xval = np.load('D://deep learning//Mini Project//x_data.npy')