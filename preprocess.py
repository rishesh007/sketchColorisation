# import tensorflow as tf
import numpy as np
import cv2
import os

# dataset = 'sketchColorisation/Images/' 
# graystore = 'sketchColorisation/grayScale/'
# rgbstore = 'sketchColorisation/colored/'
# validation = 'sketchColorisation/validation/'


dataset = 'Images/' 
graystore = 'grayScale/'
rgbstore = 'colored/'
validation = 'validation/'

samples = len(os.listdir(dataset))
print((samples))

x_shape = 512
y_shape = 512

rgb = np.zeros((samples, x_shape, y_shape, 3))
gray = np.zeros((samples, x_shape, y_shape, 1))

for i, image in enumerate(os.listdir(dataset)[:samples]):
    I = cv2.imread(dataset+image)
    # print(image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb[i] = I
    gray[i] = J

for j in range(samples):
    cv2.imwrite(validation + str(j) +'.jpg', rgb[j])
    cv2.imwrite(rgbstore + str(j) +'.jpg', rgb[j])
    cv2.imwrite(graystore + str(j) +'.jpg', gray[j])

print("yoyo")