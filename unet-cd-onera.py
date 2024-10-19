#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
from matplotlib import pyplot as plt
import tifffile as tiff
import glob
import os
import cv2
import tensorflow as tf
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Activation, MaxPool2D, Concatenate
from keras import optimizers
from keras.optimizers import Adam
from keras.metrics import MeanIoU
from keras import backend as K
from sklearn import metrics
# from google.colab import drive

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.filters import threshold_otsu, threshold_multiotsu
from cv2 import adaptiveThreshold


# In[9]:


kernel_initializer =  'he_uniform'  # also try 'he' but model not converging...

def conv_block(inputs, filter_count, pool=True, batchnorm = True):

    #first layer
    x = Conv2D(filter_count, 3, padding = 'same', kernel_initializer = 'he_uniform')(inputs)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #second layer
    x = Conv2D(filter_count, 3, padding = 'same', kernel_initializer = 'he_uniform')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if pool == True:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x



def deconv_block(inputs, concat_layer, filter_count, pool = False):
    u = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(inputs)
    c = Concatenate()([u, concat_layer])
    x = conv_block(c, filter_count, pool = pool, batchnorm = True)
    return u, c, x

def make_me_a_unet(shape, num_classes):

    inputs = Input(shape) # 768 x 1152

    # Downsampling side of the UNET i.e. the encoder !

    x1, p1 = conv_block(inputs, 16, pool=True, batchnorm=True)
    x2, p2 = conv_block(p1, 32, pool=True, batchnorm=True)
    x3, p3 = conv_block(p2, 64, pool=True, batchnorm=True)
    x4, p4 = conv_block(p3, 128, pool=True, batchnorm=True)
    b = conv_block(p4, 256, pool=False, batchnorm=True)

    # Upsampling side of the UNET i.e the decoder !

    u1, c1, x5 = deconv_block(b, x4, 128)
    u2, c2, x6 = deconv_block(x5, x3, 64)
    u3, c3, x7 = deconv_block(x6, x2, 32)
    u4, c4, x8 = deconv_block(x7, x1, 16)

    # The output layer

    output = Conv2D(num_classes, 1, padding='same', activation='sigmoid')(x8)

    #softmax for multiclass classification, num_classes = 23 !

    return Model(inputs, output)




model = make_me_a_unet((IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), 1)
model.summary()


# In[10]:


image_number = random.randint(0, len(X_train))
print(image_number)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256,3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()


# In[11]:


get_ipython().run_cell_magic('time', '', "seed=24\n\nimg_data_gen_args = dict(rotation_range=90,\n                     width_shift_range=0.3,\n                     height_shift_range=0.3,\n                     horizontal_flip=True,\n                     vertical_flip=True,\n                     shear_range=0.5,\n                     fill_mode='reflect')\n\nmask_data_gen_args = dict(rotation_range=90,\n                     width_shift_range=0.3,\n                     height_shift_range=0.3,\n                     horizontal_flip=True,\n                     vertical_flip=True,\n                     shear_range=0.5,\n                     fill_mode='reflect',\n                     preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again.\nX_train=X_train.reshape(582, 256, 256, 3)\nX_test=X_test.reshape(195, 256, 256, 3)\nimage_data_generator = ImageDataGenerator(**img_data_gen_args)\nimage_data_generator.fit(X_train, augment=True, seed=seed)\nimage_generator = image_data_generator.flow(X_train, seed=seed)\nvalid_img_generator = image_data_generator.flow(X_test, seed=seed)\n")


# In[12]:


get_ipython().run_cell_magic('time', '', '\nmask_data_generator = ImageDataGenerator(**mask_data_gen_args)\nmask_data_generator.fit(y_train, augment=True, seed=seed)\nmask_generator = mask_data_generator.flow(y_train, seed=seed)\nvalid_mask_generator = mask_data_generator.flow(y_test, seed=seed)\n')


# In[13]:


def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

my_generator = my_image_mask_generator(image_generator, mask_generator)

validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)



# In[ ]:


get_ipython().run_cell_magic('time', '', '# model = get_model()\nfrom tensorflow import keras\n#@title Select parameters {run: "auto"}\n\noptimizer = \'nag\' #@param ["adam" , "momentum" , "rmsprop" , "adagrad", "nag"] {type :"string"}\nLearning_rate = 0.01 #@param {type:"number"}\nLR = float(Learning_rate)\nif (optimizer=="adagrad"):\n  opt = keras.optimizers.Adagrad(learning_rate=LR)\nif (optimizer=="adam"):\n  opt = keras.optimizers.Adam(learning_rate=LR)\nif (optimizer=="RMSprop"):\n  opt = keras.optimizers.RMSProp(learning_rate=LR)\nif (optimizer=="momentum"):\n  opt = keras.optimizers.SGD(learning_rate=LR,momentum=0.9)\nif (optimizer=="nag"):\n  opt = keras.optimizers.SGD(learning_rate=LR,momentum=0.9,nesterov=True)\n\nbatch_size = 64\nsteps_per_epoch = 3*(len(X_train))//batch_size\nmodel.compile(optimizer= opt, loss= \'binary_crossentropy\', metrics=[\'acc\'])\ncallback = tf.keras.callbacks.EarlyStopping(monitor=\'loss\', patience=10)\nhistory = model.fit(my_generator,\n                    validation_data = validation_datagen,\n                    steps_per_epoch = steps_per_epoch,\n                    validation_steps = steps_per_epoch,\n                    epochs=200)\n')


# In[ ]:


#plot the training and validation accuracy and loss at each epoch
# batch_size*steps_per_epoch*len(loss)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
# test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
print(test_img_number)
prediction = model.predict(test_img_input)
prediction = prediction> threshold_otsu(prediction)
prediction=prediction[0,:,:,0]
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()

