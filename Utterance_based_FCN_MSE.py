# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:24:51 2018

!!! This code is recomended to be implemented with Keras version 1.1.0 (Since BatchNormalization seems to be modified in the future versions) !!!!!!

!!! If you find that the high frequency regions of enhanced speech are missing, please train FCN for more epochs (although the loss may not change a lot).
     You may ovserve the high frequency regions gradually appear as shown in the .gif here:   https://jasonswfu.github.io/JasonFu.github.io/   and  https://jasonswfu.github.io/JasonFu.github.io/images/t2.gif

This code is used for FCN-based raw waveform denoising (utterance-wise, with MSE loss)

If you find this code useful in your research, please cite:
Citation: 
       [1] S.-W. Fu, Y. Tsao, X. Lu, and H. Kawai, "Raw waveform-based speech enhancement by fully convolutional networks," in Proc. APSIPA, 2017.
       [2] S.-W. Fu, Y. Tsao, X. Lu, and H. Kawai, "End-to-end waveform utterance enhancement for direct evaluation metrics optimization by fully convolutional neural networks," IEEE Transactions on Audio, Speech, and Language Processing, 2018.
Contact:
       Szu-Wei Fu
       jasonfu@citi.sinica.edu.tw
       Academia Sinica, Taipei, Taiwan
       
@author: Jason
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from scipy.io import wavfile

import scipy.io
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
import theano
import theano.tensor as T
random.seed(999)

Num_traindata=20000
epoch=40
batch_size=1
max_input_audio_length=7 # In a 12GB RAM TITAN X GPU, with the current FCN structure, the maximun input audio length without OOM is roughly 7s.   
  
  
def shuffle_list(x_old,index):
    x_new=[x_old[i] for i in index]
    return x_new    
     
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.
    


def train_data_generator(noisy_list, clean_path):
	index=0
	while True:
         #noisy, rate  = librosa.load(noisy_list[index],sr=16000) 
         rate, noisy = wavfile.read(noisy_list[index])
         while noisy.shape[0]/16000.>max_input_audio_length: # Audio length <7s or OOM. Read next utterance. 
             index += 1
             if index == len(noisy_list):
                 index = 0
             rate, noisy = wavfile.read(noisy_list[index])
         
         noisy=noisy.astype('float')         
         if len(noisy.shape)==2:
             noisy=(noisy[:,0]+noisy[:,1])/2       
         noisy=noisy/np.max(abs(noisy))
         noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))
         
         #clean, rate  =librosa.load(clean_list[clean_wav_list.index(noisy_wav_list[index])],sr=16000)         
         rate, clean = wavfile.read(clean_path+noisy_list[index].split('/')[-1])
         clean=clean.astype('float')  
         if len(clean.shape)==2:
             clean=(clean[:,0]+clean[:,1])/2
         clean=clean/np.max(abs(clean))         
         clean=np.reshape(clean,(1,np.shape(clean)[0],1))
         
         
         index += 1
         if index == len(noisy_list):
             index = 0
                         
             random.shuffle(noisy_list)
          
         yield noisy, clean

def val_data_generator(noisy_list, clean_path):
	index=0
	while True:
         #noisy, rate  = librosa.load(noisy_list[index],sr=16000)       
         rate, noisy = wavfile.read(noisy_list[index])
         noisy=noisy.astype('float')         
         if len(noisy.shape)==2:
             noisy=(noisy[:,0]+noisy[:,1])/2       
         noisy=noisy/np.max(abs(noisy))
         noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))
         
         #clean, rate  =librosa.load(clean_list[clean_wav_list.index(noisy_wav_list[index])],sr=16000)         
         rate, clean = wavfile.read(clean_path+noisy_list[index][noisy_list[index].index('dB')+2:])
         clean=clean.astype('float')  
         if len(clean.shape)==2:
             clean=(clean[:,0]+clean[:,1])/2
         clean=clean/np.max(abs(clean))         
         clean=np.reshape(clean,(1,np.shape(clean)[0],1))
         
         
         index += 1
         if index == len(noisy_list):
             index = 0
          
         yield noisy, clean 

# Data Path: change to your path!
######################### Training data #########################
Train_Noisy_lists = get_filepaths("/mnt/hd-02/avse/training/noisy") # Please change to your path
Train_Clean_paths = "/mnt/hd-02/avse/training/clean/"                   # Please change to your path

   
# data_shuffle
random.shuffle(Train_Noisy_lists)

Train_Noisy_lists=Train_Noisy_lists[0:Num_traindata]      # Only use subset of training data

steps_per_epoch = (Num_traindata)//batch_size
######################### Test_set #########################
Test_Noisy_lists  = get_filepaths("/mnt/hd-02/avse/testing/noisy") # Please change to your path
Test_Clean_paths = "/mnt/hd-02/avse/testing/clean/"                    # Please change to your path
                      
Num_testdata=len(Test_Noisy_lists)   

         
start_time = time.time()

print 'model building...'

model = Sequential()


model.add(Convolution1D(30, 55, border_mode='same', input_shape=(None,1)))
model.add(BatchNormalization(mode=2,axis=-1))   # Instance Normalization. Because of batch size=1.
model.add(LeakyReLU())
#model.add(Dropout(0.06))

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())
#model.add(Dropout(0.06))

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())
#model.add(Dropout(0.06))

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())
#model.add(Dropout(0.06))

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())
#model.add(Dropout(0.06))


model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())
#model.add(Dropout(0.06))

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())
#model.add(Dropout(0.06))

model.add(Convolution1D(1, 55,  border_mode='same'))
model.add(Activation('tanh'))

model.compile(loss='mse', optimizer='adam')
    
with open('FCNN_MSE.json','w') as f:    # save the model
    f.write(model.to_json()) 
checkpointer = ModelCheckpoint(filepath='FCNN_MSE.hdf5', verbose=1, save_best_only=True, mode='min')  

print 'training...'
g1 = train_data_generator(Train_Noisy_lists, Train_Clean_paths)
g2 = val_data_generator  (Test_Noisy_lists, Test_Clean_paths)

hist=model.fit_generator(g1,	
                         samples_per_epoch=Num_traindata, 
				  nb_epoch=epoch, 
				  verbose=1,
                         validation_data=g2,
                         nb_val_samples=Num_testdata,
                         max_q_size=1, 
                         nb_worker=1,
                         pickle_safe=True,
                         callbacks=[checkpointer]
                         )                            					

print 'De-noising...'
maxv = np.iinfo(np.int16).max 
for path in Test_Noisy_lists: # Ex: /mnt/hd-02/avse/testing/noisy/engine/1dB/1.wav
    S=path.split('/') 
    noise=S[-3]
    dB=S[-2]
    wave_name=S[-1]
    
    rate, noisy = wavfile.read(path)
    noisy=noisy.astype('float')
    if len(noisy.shape)==2:
        noisy=(noisy[:,0]+noisy[:,1])/2             
    noisy=noisy/np.max(abs(noisy))
    noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))
    enhanced=np.squeeze(model.predict(noisy, verbose=0, batch_size=batch_size))
    enhanced=enhanced/np.max(abs(enhanced))
    librosa.output.write_wav(os.path.join("FCN_enhanced_MSE",noise, dB, wave_name), (enhanced* maxv).astype(np.int16), 16000)

# plotting the learning curve
TrainERR=hist.history['loss']
ValidERR=hist.history['val_loss']
print ('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
print 'drawing the training process...'
plt.figure(2)
plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
plt.xlim([1,epoch])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error')
plt.grid(True)
plt.show()
plt.savefig('Learning_curve_FCN_MSE.png', dpi=150)


end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))

