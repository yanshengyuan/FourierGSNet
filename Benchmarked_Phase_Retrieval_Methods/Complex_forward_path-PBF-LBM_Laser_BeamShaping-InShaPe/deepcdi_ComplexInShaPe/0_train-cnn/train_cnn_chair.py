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

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
  tf.config.set_visible_devices(gpus[0], 'GPU')
else:
  print("No GPU available, using CPU.")

beamshape = 'bench'
num_training_samples = 10000

WEIGHTS_PATH  = './0_train-cnn/output/weights'
NUM_EPOCHS    = 30
LEARNING_RATE = 0.0002
DR_RATE       = 0.2

trainx_folder = '../InShaPe_dataset/dense'+beamshape+'30k_pre/training_set/intensity/npy'
trainy_folder = '../InShaPe_dataset/dense'+beamshape+'30k_pre/training_set/phase/npy'
valx_folder = '../InShaPe_dataset/dense'+beamshape+'30k_pre/test_set/intensity/npy'
valy_folder = '../InShaPe_dataset/dense'+beamshape+'30k_pre/test_set/phase/npy'

if(beamshape=='bench'):
    beamshape = 'chair'

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
  train_x = np.empty((num_training_samples, 427, 427))
  train_y = np.empty((num_training_samples, 427, 427))
  val_x = np.empty((3000, 427, 427))
  val_y = np.empty((3000, 427, 427))
  
  trainx_files = [f for f in os.listdir(trainx_folder) if f.endswith('.npy')]
  trainx_files.sort()
  trainy_files = [f for f in os.listdir(trainy_folder) if f.endswith('.npy')]
  trainy_files.sort()
  valx_files = [f for f in os.listdir(valx_folder) if f.endswith('.npy')]
  valx_files.sort()
  valy_files = [f for f in os.listdir(valy_folder) if f.endswith('.npy')]
  valy_files.sort()
  
  for i in range(num_training_samples):
    xfile_path = os.path.join(trainx_folder, trainx_files[i])
    yfile_path = os.path.join(trainy_folder, trainy_files[i])
    xarray = np.load(xfile_path)
    yarray = np.load(yfile_path)
    
    train_x[i] = xarray
    train_y[i] = yarray
    
    print(f"{trainx_files[i]} loaded with shape {xarray.shape}")
    print(f"{trainy_files[i]} loaded with shape {yarray.shape}")
    
  for i in range(len(valx_files)):
    xfile_path = os.path.join(valx_folder, valx_files[i])
    yfile_path = os.path.join(valy_folder, valy_files[i])
    xarray = np.load(xfile_path)
    yarray = np.load(yfile_path)
    
    val_x[i] = xarray
    val_y[i] = yarray
    
    print(f"{valx_files[i]} loaded with shape {xarray.shape}")
    print(f"{valy_files[i]} loaded with shape {yarray.shape}")
  
  train_x = train_x[..., np.newaxis]
  train_y = train_y[..., np.newaxis]
  val_x = val_x[..., np.newaxis]
  val_y = val_y[..., np.newaxis]
  #print(train_x.shape)
  #print(train_y.shape)
  #print(val_x.shape)
  #print(val_y.shape)
  
  #plt.imsave("input_train.png", train_x[0].squeeze(), cmap='gray')
  #plt.imsave("gt_train.png", train_y[0].squeeze(), cmap='gray')
  #plt.imsave("input_val.png", val_x[0].squeeze(), cmap='gray')
  #plt.imsave("gt_val.png", val_y[0].squeeze(), cmap='gray')

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
  for i in range(len(valx_files)):
    xfile_path = os.path.join(valx_folder, valx_files[i])
    yfile_path = os.path.join(valy_folder, valy_files[i])
    xarray = np.load(xfile_path)
    yarray = np.load(yfile_path)
    
    test_sample = np.empty((1, 427, 427))
    test_sample[0] = xarray
    test_sample = test_sample[..., np.newaxis]
    
    start = time.perf_counter()
    pred = model.predict(test_sample)
    end = time.perf_counter()
    inf_time_list.append(end-start)
    
    pred = pred.squeeze()
    test_sample = test_sample.squeeze()
    np.save('deepcdi_'+beamshape+'/Phi_pred/npy/'+str(i)+'_phi_pred.npy' ,pred)
    plt.imsave('deepcdi_'+beamshape+'/Phi_pred/img/'+str(i)+'_phi_pred.png' ,pred, cmap='gray')
    np.save('deepcdi_'+beamshape+'/Phi_gt/npy/'+str(i)+'_phi_gt.npy' ,yarray)
    plt.imsave('deepcdi_'+beamshape+'/Phi_gt/img/'+str(i)+'_phi_gt.png' ,yarray, cmap='gray')
    np.save('deepcdi_'+beamshape+'/I_gt/npy/'+str(i)+'_I_gt.npy' ,test_sample)
    plt.imsave('deepcdi_'+beamshape+'/I_gt/img/'+str(i)+'_I_gt.png' ,test_sample, cmap='gray')
    print('inference '+str(i))
  inf_time = np.array(inf_time_list)
  np.save("inf_time.npy", inf_time)

if __name__ == "__main__":
  main()

