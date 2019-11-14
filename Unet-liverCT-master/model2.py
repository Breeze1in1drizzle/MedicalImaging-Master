import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 1})))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def BatchActivate(x):
    # 规范化
    x = BatchNormalization()(x)
    #激活函数
    x = Activation('relu')(x)
    return x
def residual_block(con,start):
    con_after = Conv2D(start,(3,3),activation =None, padding = 'same', kernel_initializer = 'he_normal')(con)
    con_after = BatchActivate(con_after)
    con_after = Add()([con,con_after])
    con_after = BatchActivate(con_after)
    return con_after

def unet(pretrained_weights = None, input_size = (512, 512, 1)):

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation =None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchActivate(conv1)
    conv1 = Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchActivate(conv1)
    conv1 = residual_block(conv1,64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


    conv2 = Conv2D(128, 3, activation =None, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchActivate(conv2)
    conv2 = Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchActivate(conv2)
    conv2 = residual_block(conv2,128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchActivate(conv3)
    conv3 = Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchActivate(conv3)
    conv3 = residual_block(conv3,256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchActivate(conv4)
    conv4 = Conv2D(512, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchActivate(conv4)
    conv4 = residual_block(conv4,512)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchActivate(conv5)
    conv5 = Conv2D(1024, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchActivate(conv5)
    conv5 = residual_block(conv5,1024)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    up6 = BatchActivate(up6)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchActivate(conv6)
    conv6 = Conv2D(512, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchActivate(conv6)
    conv6 = residual_block(conv6,512)



    up7 = Conv2D(256, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchActivate(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchActivate(conv7)
    conv7 = Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchActivate(conv7)
    conv7 = residual_block(conv7,256)


    up8 = Conv2D(128, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchActivate(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchActivate(conv8)
    conv8 = Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchActivate(conv8)
    conv8 = residual_block(conv8,128)


    up9 = Conv2D(64, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = BatchActivate(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation =None, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchActivate(conv9)
    conv9 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchActivate(conv9)
    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = residual_block(conv9,64)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


