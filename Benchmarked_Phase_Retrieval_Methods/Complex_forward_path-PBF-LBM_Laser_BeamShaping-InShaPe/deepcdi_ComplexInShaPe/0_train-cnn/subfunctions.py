#Implemented by Dillan J. Chang: https://github.com/dillanchang/deep-phase-imaging

import numpy as np
from tensorflow.keras import layers, models
import tensorflow as tf

def imresize_big(img, factor):
  img_big = np.zeros((img.shape[0]*factor,img.shape[1]*factor))
  for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
      x = i*factor
      y = j*factor
      for a in range(0,factor):
        for b in range(0,factor):
          img_big[x+a,y+b] = img[i,j]
  return img_big

def add_residual_block(output_dim,input_layer):
  layer = input_layer
  layer = layers.BatchNormalization()(layer)
  layer = layers.ReLU()(layer)
  layer = layers.Conv2D(output_dim, (3,3), padding='same')(layer)
  layer = layers.BatchNormalization()(layer)
  layer = layers.ReLU()(layer)
  layer = layers.Conv2D(output_dim, (3,3), padding='same')(layer)
  residual_layer = layers.Conv2D(output_dim, (3,3), padding='same')(input_layer)
  layer = layers.Add()([layer,residual_layer])
  return layer

def add_residual_downsampling_block(output_dim,input_layer):
  layer = input_layer
  layer = layers.BatchNormalization()(layer)
  layer = layers.ReLU()(layer)
  layer = layers.Conv2D(output_dim, (3,3), strides=(2,2), padding='same')(layer)
  layer = layers.BatchNormalization()(layer)
  layer = layers.ReLU()(layer)
  layer = layers.Conv2D(output_dim, (3,3), padding='same')(layer)
  residual_layer = layers.Conv2D(output_dim, (3,3), strides=(2,2), padding='same')(input_layer)
  layer = layers.Add()([layer,residual_layer])
  return layer

def add_down_residual_layer(output_dim,input_layer):
  layer = add_residual_downsampling_block(output_dim,input_layer)
  layer = add_residual_block(output_dim,layer)
  return layer

def add_residual_upsampling_block(output_dim,input_layer):
  layer = input_layer
  layer = layers.BatchNormalization()(layer)
  layer = layers.ReLU()(layer)
  layer = layers.Conv2DTranspose(output_dim, (3,3), strides=(2,2), padding='same')(layer)
  layer = layers.BatchNormalization()(layer)
  layer = layers.ReLU()(layer)
  layer = layers.Conv2DTranspose(output_dim, (3,3), padding='same')(layer)
  residual_layer = layers.Conv2DTranspose(output_dim, (3,3), strides=(2,2), padding='same')(input_layer)
  layer = layers.Add()([layer,residual_layer])
  return layer

def add_up_residual_layer(output_dim,input_layer):
  layer = add_residual_upsampling_block(output_dim,input_layer)
  layer = add_residual_block(output_dim,layer)
  return layer

def match_shape(x, skip):
    x_shape = x.shape[1:3]
    skip_shape = skip.shape[1:3]

    pad_h = x_shape[0] - skip_shape[0]
    pad_w = x_shape[1] - skip_shape[1]

    if pad_h != 0 or pad_w != 0:
        skip = layers.ZeroPadding2D(padding=((0, pad_h), (0, pad_w)))(skip)
    return skip

def create_model(dr_rate):
  input_layer = layers.Input((427,427,1))
  cnn   = input_layer

  cnn_a = add_residual_block(16,cnn)
  conn1 = layers.Dropout(rate=dr_rate)(cnn_a)
  cnn   = layers.Dropout(rate=dr_rate)(cnn_a)

  cnn_b = add_down_residual_layer(32,cnn)
  conn2 = layers.Dropout(rate=dr_rate)(cnn_b)
  cnn   = layers.Dropout(rate=dr_rate)(cnn_b)

  cnn_c = add_down_residual_layer(64,cnn)
  conn3 = layers.Dropout(rate=dr_rate)(cnn_c)
  cnn   = layers.Dropout(rate=dr_rate)(cnn_c)

  cnn_d = add_down_residual_layer(128,cnn)
  conn4 = layers.Dropout(rate=dr_rate)(cnn_d)
  cnn   = layers.Dropout(rate=dr_rate)(cnn_d)

  cnn   = add_down_residual_layer(256,cnn)
  cnn   = layers.Dropout(rate=dr_rate)(cnn)

  cnn   = add_up_residual_layer(256,cnn)
  cnn   = layers.Dropout(rate=dr_rate)(cnn)
  conn4 = match_shape(cnn, conn4)
  cnn   = layers.Concatenate(axis=3)([cnn,conn4])

  cnn   = add_up_residual_layer(128,cnn)
  cnn   = layers.Dropout(rate=dr_rate)(cnn)
  conn3 = match_shape(cnn, conn3)
  cnn   = layers.Concatenate(axis=3)([cnn,conn3])

  cnn   = add_up_residual_layer(64,cnn)
  cnn   = layers.Dropout(rate=dr_rate)(cnn)
  conn2 = match_shape(cnn, conn2)
  cnn   = layers.Concatenate(axis=3)([cnn,conn2])

  cnn   = add_up_residual_layer(32,cnn)
  cnn   = layers.Dropout(rate=dr_rate)(cnn)
  conn1 = match_shape(cnn, conn1)
  cnn   = layers.Concatenate(axis=3)([cnn,conn1])
  
  cnn = layers.Lambda(lambda x: tf.image.resize(x, (427, 427)))(cnn)
  cnn   = add_residual_block(32,cnn)
  cnn   = add_residual_block(1,cnn)

  model = models.Model(input_layer,cnn)
  return model

