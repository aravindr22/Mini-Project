# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:45:14 2021

@author: Aravind
"""

import os
f='D://deep learning//Mini Project//images//'
extensions={'jp':'jpg','jpe':'jpeg','gi':'gif','mp':'mp4'}
c = ['chicken_wings',
 'donuts',
 'french_fries',
 'fried_rice',
 'hamburger',
 'hot_dog',
 'ice_cream',
 'nachos',
 'omelette',
 'pizza',
 'samosa',
 'sushi']

for j in c:
    folder_path = f + j
    for i in os.listdir(folder_path):
        paths=i.split('.') #==== Split the string based on the parameter
        if paths[-1] == 'jpg':
            #print("asd")
            os.rename(os.path.join(folder_path,i),os.path.join(folder_path,paths[0]+".jpeg")) 