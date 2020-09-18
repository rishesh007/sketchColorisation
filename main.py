import os
import cv2
import numpy as np
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.losses import *
fixed_seed_num = 1234
np.random.seed(fixed_seed_num)
tf.random.set_seed(fixed_seed_num)

x_shape = 512
y_shape = 512

def generator_model(x_shape,y_shape):
    
    # encoder
    generator_input = tf.keras.Input(batch_shape=(None,x_shape,y_shape, 1), name='generator_input')
    generator_input_normalized = tf.keras.layers.BatchNormalization()(generator_input)
    
    conv1_32 = tf.keras.layers.Conv2D(16,kernel_size=(3,3),strides=(1,1),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(generator_input)
    conv1_32 = tf.keras.layers.BatchNormalization()(conv1_32)
    
    conv2_64 = tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv1_32)
    conv2_64 = tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv2_64)    
    conv2_64 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding="same")(conv2_64)
    conv2_64 = tf.keras.layers.BatchNormalization()(conv2_64)
    
    conv3_128 = tf.keras.layers.Conv2D(64,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv2_64)
    conv3_128 = tf.keras.layers.Conv2D(64,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv3_128)
    conv3_128 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding="same")(conv3_128)
    conv3_128 = tf.keras.layers.BatchNormalization()(conv3_128)
    
    conv4_256 = tf.keras.layers.Conv2D(128,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv3_128)
    conv4_256 = tf.keras.layers.Conv2D(128,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv4_256)
    conv4_256 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding="same")(conv4_256)
    conv4_256 = tf.keras.layers.BatchNormalization()(conv4_256)
    
    conv5_512 = tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv4_256)
    conv5_512 = tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv5_512)
    conv5_512 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding="same")(conv5_512)
    conv5_512 = tf.keras.layers.BatchNormalization()(conv5_512)
    
    conv6_512 = tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv5_512)
    conv6_512 = tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv5_512)
    conv6_512 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding="same")(conv6_512)
    conv6_512 = tf.keras.layers.BatchNormalization()(conv6_512)
    
    conv7_512 = tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv6_512)
    conv7_512 = tf.keras.layers.BatchNormalization()(conv7_512)
    
    # decoder
    conv8_512 = tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv7_512)
    conv8_512 = tf.keras.layers.BatchNormalization(axis=1)(conv8_512)
    
    deconv9_512 = tf.keras.layers.Conv2DTranspose(512,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2),kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv8_512)
    deconv9_512 = tf.keras.layers.BatchNormalization()(deconv9_512)
    deconv9_512 = tf.keras.layers.Concatenate()([deconv9_512,conv5_512])
    deconv9_512 = tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(deconv9_512)
    deconv9_512 = tf.keras.layers.BatchNormalization()(deconv9_512)
    
    deconv10_256 = tf.keras.layers.Conv2DTranspose(256,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2),kernel_regularizer=tf.keras.regularizers.l2(0.001))(deconv9_512)
    deconv10_256 = tf.keras.layers.BatchNormalization()(deconv10_256)
    deconv10_256 = tf.keras.layers.Concatenate()([deconv10_256,conv4_256])
    deconv10_256 = tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(deconv10_256)
    deconv10_256 = tf.keras.layers.BatchNormalization()(deconv10_256)
    
    deconv11_128 = tf.keras.layers.Conv2DTranspose(128,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2),kernel_regularizer=tf.keras.regularizers.l2(0.001))(deconv10_256)
    deconv11_128 = tf.keras.layers.Concatenate()([deconv11_128,conv3_128])
    deconv11_128 = tf.keras.layers.Conv2D(128,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(deconv11_128)
    
    deconv12_64 = tf.keras.layers.Conv2DTranspose(64,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2),kernel_regularizer=tf.keras.regularizers.l2(0.001))(deconv11_128)
    deconv12_64 = tf.keras.layers.Concatenate()([deconv12_64,conv2_64])
    deconv12_64 = tf.keras.layers.Conv2D(64,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(deconv12_64)
    
    deconv13_32 = tf.keras.layers.Conv2DTranspose(32,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2),kernel_regularizer=tf.keras.regularizers.l2(0.001))(deconv12_64)
    deconv13_32 = tf.keras.layers.Concatenate()([deconv13_32,conv1_32])
    deconv13_32 = tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(deconv13_32)
    
    deconv14_16 = tf.keras.layers.Conv2DTranspose(16,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(deconv13_32)
    deconv14_16 = tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(deconv14_16)
    
    output = tf.keras.layers.Conv2D(3,kernel_size=(1,1),padding='same',activation='relu')(deconv14_16)
    
    model = tf.keras.Model(inputs=generator_input,outputs=output)
    
    return model

def discriminator_model(x_shape,y_shape):
    
    generator_input = tf.keras.Input(batch_shape=(None, x_shape, y_shape, 1), name='generator_output')
    generator_output = tf.keras.Input(batch_shape=(None, x_shape, y_shape, 3), name='generator_input')
    
    input1 = tf.keras.layers.BatchNormalization()(generator_input)
    input2 = tf.keras.layers.BatchNormalization()(generator_output)
    
    convi = tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='elu',padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001))(generator_input)
    convi = tf.keras.layers.BatchNormalization()(convi)
    
    convo = tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='elu',padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001))(generator_output)
    convo = tf.keras.layers.BatchNormalization()(convo)

    
    convi = tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='elu',padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001))(convi)
    convi = tf.keras.layers.BatchNormalization()(convi)
    
    convo = tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='elu',padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001))(convo)
    convo = tf.keras.layers.BatchNormalization()(convo)

    
    convi = tf.keras.layers.Conv2D(64,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001))(convi)
    convo = tf.keras.layers.Conv2D(64,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001))(convo)
    
    conv = tf.keras.layers.Concatenate()([convi,convo])
    conv = tf.keras.layers.BatchNormalization()(conv)
    
    conv = tf.keras.layers.Conv2D(128,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    
    conv = tf.keras.layers.Conv2D(128,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    
    conv = tf.keras.layers.Conv2D(128,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    
    conv = tf.keras.layers.Conv2D(256,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    
    conv = tf.keras.layers.Conv2D(256,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    
    conv = tf.keras.layers.Flatten()(conv)
    conv = tf.keras.layers.Dropout(0.5)(conv)
    
    conv = tf.keras.layers.Dense(100,activation='elu')(conv)
    conv = tf.keras.layers.Dropout(0.5)(conv)
    
    output = tf.keras.layers.Dense(1,activation='sigmoid')(conv)
    
    model = tf.keras.Model(inputs=([generator_input,generator_output]),outputs=[output])
    
    return model

def cGAN_model(generator,discriminator):
    
    discriminator.trainable = False
    model = tf.keras.Model(inputs=generator.inputs,outputs=[discriminator([generator.input,generator.output]), generator.output])
    
    return model

def custom_loss(y_true,y_pred):
    cosine = tf.keras.losses.cosine_proximity(y_true,y_pred)
    mle = tf.keras.losses.MAE(y_true, y_pred)
    l = (cosine)+mle
    
    return l

def custom_loss_2(y_true,y_pred):
    cosine = cosine_similarity(y_true,y_pred,axis=-1)
    mse = MSE(y_true, y_pred)
    mle = MAE(y_true, y_pred)
    l = (cosine+1)*mse+mle
    return l

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

# get validation dataset 
for i, image in enumerate(os.listdir(val_data)[:val_samples]):
    I = cv2.imread(val_data+image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb_val[i] = I; gray_val[i] = J


datagen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2,fill_mode='wrap',horizontal_flip=True,vertical_flip=True,rotation_range=15)
datagen.fit(rgb)

# train the cGAN model with specified number of epochs
epochs = 5000
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
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    if e%100 == 0:
        cGAN.save_weights(store+str(e)+'.h5') 
    gen_image_val = gen.predict(gray_val, batch_size=8)
    if e%100 == 0: 
        for j in range(val_samples):
            if not os.path.exists(store2+str(j)+'/'):
                os.mkdir(store2+str(j)+'/')
            cv2.imwrite(store2+str(j)+'/'+str(e)+'.jpg', gen_image_val[j])


