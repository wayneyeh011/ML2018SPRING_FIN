import os
import shutil
import platform
import numpy as np
np.random.seed(2002)  # base 1001; others 2002, 3003, 4004, 5005

import pandas as pd
import scipy
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D,
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate, LSTM, Flatten,
                          Lambda, Conv2DTranspose, BatchNormalization, Reshape,
                          Convolution2D, GlobalMaxPool1D, MaxPool2D, Activation)
from keras.utils import Sequence, to_categorical
from sklearn.cross_validation import StratifiedKFold
import keras.backend as K 
import librosa


## NOTE
#  (1) This .py turn a sound example into 2darray (mfcc transform)
#  (2) One should check the input dir, some setting (like duration) and output dir
#  (ref) https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data/comments



## Setting
my_dur = 2

train_csv_path='../input/train.csv'
train_audio_folder_path='../input/audio_train/'
test_csv_path='../input/sample_submission.csv'
test_audio_folder_path='../input/audio_test/'
#save_log_folder='ignore/logs/'



## Setting
class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001,
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)


## Importing
train = pd.read_csv(train_csv_path)
test = pd.read_csv(test_csv_path)
print("Number of training examples=", train.shape[0], "  Number of classes=", len(train.label.unique()))

LABELS = list(train.label.unique())
print ('Some example of label',LABELS[:5])

label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])
print ('\n',train.head())
print ('\n',test.head())



## Preparing 
def prepare_data(df, config, data_dir):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    for i, fname in enumerate(df.index):
        print(fname)
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

        # Random offset / Padding
        if len(data) > input_length:   # data longer
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:                          # data shorter --- pad on LHS and RHS with random size 
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        data = np.expand_dims(data, axis=-1)
        X[i,] = data
    return X



## To mfcc (2d array)
config = Config(sampling_rate=44100, audio_duration=my_dur, n_folds=10, 
                learning_rate=0.001, use_mfcc=True, n_mfcc=40)
X_train = prepare_data(train, config, '../input/audio_train/')
print ('Train Done')
X_test = prepare_data(test, config, '../input/audio_test/')
print ('Test Done')
#y_train = to_categorical(train.label_idx, num_classes=config.n_classes)
#print ('Shape of Train, Test and Y', X_train.shape, X_test.shape, y_train.shape) 
print ('Shape of Train, Test', X_train.shape, X_test.shape)

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std
print ('Normalization Done')


## Saving
np.save('../input/X_train_dur' + str(my_dur) + '.npy' ,X_train)
np.save('../input/X_test_dur' + str(my_dur) + '.npy',X_test)
#np.save('../input/y_train.npy',y_train)






