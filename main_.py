import os
import cv2
import numpy as np
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.losses import *
from loss import custom_loss_2
from model import generator_model, discriminator_model, cGAN_model

fixed_seed_num = 1234
np.random.seed(fixed_seed_num)
tf.random.set_seed(fixed_seed_num)

x_shape = 512
y_shape = 512

def train(gen,disc,cGAN,gray,rgb,gray_val,rgb_val,batch):
    samples = len(rgb)
    gen_image = gen.predict(gray, batch_size=16)   
    gen_image_val = gen.predict(gray_val, batch_size=8)
    inputs = np.concatenate([gray, gray])
    outputs = np.concatenate([rgb, gen_image])
    y = np.concatenate([np.ones((samples, 1)), np.zeros((samples, 1))])
    disc.fit([inputs, outputs], y, epochs=1, batch_size=4)
    disc.trainable = False
    cGAN.fit(gray, [np.ones((samples, 1)), rgb], epochs=1, batch_size=batch,validation_data=[gray_val,[np.ones((val_samples,1)),rgb_val]])
    disc.trainable = True

gen = generator_model(x_shape,y_shape)

disc = discriminator_model(x_shape,y_shape)

cGAN = cGAN_model(gen, disc)
# cGAN.load_weights('sketchColorisation/result/store/9950.h5')

disc.compile(loss=['binary_crossentropy'], optimizer=tf.keras.optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])

cGAN.compile(loss=['binary_crossentropy',custom_loss_2], loss_weights=[5, 100], optimizer=tf.keras.optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))

dataset = 'sketchColorisation/Images/' 
graystore = 'sketchColorisation/grayScale/'
rgbstore = 'sketchColorisation/colored/'
val_data = 'sketchColorisation/validation/'
store = 'sketchColorisation/result/store/'
store2 = 'sketchColorisation/result/store2/'

# samples = len(os.listdir(dataset))
# val_samples = len(os.listdir(val_data))
samples = 8
val_samples = 8

rgb = np.zeros((samples, x_shape, y_shape, 3))
gray = np.zeros((samples, x_shape, y_shape, 1))
rgb_val = np.zeros((val_samples, x_shape, y_shape, 3))
gray_val = np.zeros((val_samples, x_shape, y_shape, 1))
y_train = np.zeros((samples, 1))

for i, image in enumerate(os.listdir(dataset)[:samples]):
    I = cv2.imread(dataset+image)
    # print(image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb[i] = I
    gray[i] = J

for i, image in enumerate(os.listdir(val_data)[:val_samples]):
    I = cv2.imread(val_data+image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb_val[i] = I; gray_val[i] = J


datagen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2,fill_mode='wrap',horizontal_flip=True,vertical_flip=True,rotation_range=15)
datagen.fit(rgb)

epochs = 1
device_name = '/device:GPU:0'

with tf.device(device_name):
  for e in range(epochs+1):
    batches = 0
    print('Epoch', e)
    for x_batch, y_batch in datagen.flow(rgb, y_train, batch_size=samples):
        for i in range(len(x_batch)):
            gray[i] = cv2.cvtColor(x_batch[i], cv2.COLOR_BGR2GRAY).reshape((x_shape, y_shape, 1))
        params = (gen, disc, cGAN, gray, x_batch, gray_val, rgb_val, 1)
        train(*params)
        batches += 1
        if batches >= 1:
            break
    if e%100 == 0:
        cGAN.save_weights(store+str(e)+'.h5') 
    gen_image_val = gen.predict(gray_val, batch_size=8)
    if e%1000 == 0: 
        for j in range(val_samples):
            if not os.path.exists(store2+str(j)+'/'):
                os.mkdir(store2+str(j)+'/')
            cv2.imwrite(store2+str(j)+'/'+str(e)+'.jpg', gen_image_val[j])


