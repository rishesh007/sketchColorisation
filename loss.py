import os
import cv2
import numpy as np
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.losses import *

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
