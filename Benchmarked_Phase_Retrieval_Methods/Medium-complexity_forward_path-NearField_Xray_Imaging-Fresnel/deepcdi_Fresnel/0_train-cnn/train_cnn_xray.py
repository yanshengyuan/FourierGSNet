#Implemented by Shengyuan Yan: https://github.com/yanshengyuan/FourierGSNet-Efficient_Gerchberg-Saxton_algorithm_deep_unrolling_for_all-complexity_Phase_Retrieval

import numpy as np
import matplotlib.pyplot as plt
import subfunctions as sfns
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
import os
import time
import h5py

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
  tf.config.set_visible_devices(gpus[0], 'GPU')
else:
  print("No GPU available, using CPU.")

beamshape = 'Xray'
num_training_samples = 10000

WEIGHTS_PATH  = './0_train-cnn/output/weights'
NUM_EPOCHS    = 30
LEARNING_RATE = 0.0002
DR_RATE       = 0.2

train_folder = '../PhaseGAN_data/'
val_folder = '../PhaseGAN_data/test/'

def lr_schedule(epoch, lr):
    # Decay learning rate by 10% every epoch
    if epoch % 2 == 0 and epoch > 0:
        lr = lr * 0.5
    return lr

def mse_loss(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    return K.mean(error)

def main():
  train_start = time.perf_counter()
  #--------------------------------------------------
  train_x = np.empty((num_training_samples, 256, 256))
  train_y = np.empty((num_training_samples, 256, 256))
  val_x = np.empty((3000, 256, 256))
  val_y = np.empty((3000, 256, 256))
  
  train_files = [f for f in os.listdir(train_folder) if f.endswith('.h5')]
  train_files.sort()
  val_files = [f for f in os.listdir(val_folder) if f.endswith('.h5')]
  val_files.sort()
  
  for i in range(num_training_samples):
      
    train_path = train_folder + train_files[i]
    with h5py.File(train_path, 'r') as f:
        real = f['/ph/real'][:]
        imag = f['/ph/imag'][:]
        Xray_image = f['/ph/intensity'][:]
        phase = np.arctan2(imag, real)
        
        train_x[i] = Xray_image
        train_y[i] = phase
        print(f"{train_files[i]} intensity loaded with shape {train_x[i].shape}")
        print(f"{train_files[i]} phase loaded with shape {train_y[i].shape}")
    
  for i in range(len(val_files)):
    
     val_path = val_folder + val_files[i]
     with h5py.File(val_path, 'r') as f:
         real = f['/ph/real'][:]
         imag = f['/ph/imag'][:]
         Xray_image = f['/ph/intensity'][:]
         phase = np.arctan2(imag, real)
         
         val_x[i] = Xray_image
         val_y[i] = phase
         print(f"{val_files[i]} intensity loaded with shape {val_x[i].shape}")
         print(f"{val_files[i]} phase loaded with shape {val_y[i].shape}")
  
  train_x = train_x[..., np.newaxis]
  train_y = train_y[..., np.newaxis]
  val_x = val_x[..., np.newaxis]
  val_y = val_y[..., np.newaxis]
  #print(train_x.shape)
  #print(train_y.shape)
  #print(val_x.shape)
  #print(val_y.shape)
  
  plt.imsave("input_train.png", train_x[0].squeeze(), cmap='gray')
  plt.imsave("gt_train.png", train_y[0].squeeze(), cmap='gray')
  plt.imsave("input_val.png", val_x[0].squeeze(), cmap='gray')
  plt.imsave("gt_val.png", val_y[0].squeeze(), cmap='gray')

  #--------------------------------------------------
  print("Training neural network")
  lr_scheduler = LearningRateScheduler(lr_schedule)
  model = sfns.create_model(DR_RATE)
  opt = optimizers.Adam(learning_rate=LEARNING_RATE)
  model.compile(optimizer=opt,loss=mse_loss)
  model.summary()
  checkpoints = callbacks.ModelCheckpoint('%s/{epoch:03d}.hdf5' %WEIGHTS_PATH,
    save_weights_only=False, verbose=1, save_freq="epoch")
  
  history = model.fit(train_x, train_y, shuffle=True, batch_size=2, verbose=1,
    epochs=NUM_EPOCHS, validation_data=(val_x, val_y), callbacks=[checkpoints, lr_scheduler])
  
  train_end = time.perf_counter()
  train_time = train_end - train_start
  print("total time: "+str(train_time))

  inf_time_list = []
  for i in range(len(val_files)):
    
    val_path = val_folder + val_files[i]
      
    with h5py.File(val_path, 'r') as f:
        real = f['/ph/real'][:]
        imag = f['/ph/imag'][:]
        Xray_image = f['/ph/intensity'][:]
        phase = np.arctan2(imag, real)
    
    test_sample = np.empty((1, 256, 256))
    test_sample[0] = Xray_image
    test_sample = test_sample[..., np.newaxis]
    
    start = time.perf_counter()
    pred = model.predict(test_sample)
    end = time.perf_counter()
    inf_time_list.append(end-start)
    
    pred = pred.squeeze()
    test_sample = test_sample.squeeze()
    np.save('deepcdi_'+beamshape+'/Phi_pred/npy/'+str(i)+'_phi_pred.npy' ,pred)
    plt.imsave('deepcdi_'+beamshape+'/Phi_pred/img/'+str(i)+'_phi_pred.png' ,pred, cmap='gray')
    np.save('deepcdi_'+beamshape+'/Phi_gt/npy/'+str(i)+'_phi_gt.npy', phase)
    plt.imsave('deepcdi_'+beamshape+'/Phi_gt/img/'+str(i)+'_phi_gt.png' , phase, cmap='gray')
    print('inference '+str(i))
  inf_time = np.array(inf_time_list)
  np.save("inf_time.npy", inf_time)

if __name__ == "__main__":
  main()

