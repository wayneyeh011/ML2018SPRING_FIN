import os
import shutil
import platform
import numpy as np
np.random.seed(1001)

import pandas as pd
import scipy
import sys
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
#import librosa



## NOTE
#  (1) DEMO = True ... small subsample, epoch=1, 2 folds
#  (2) One should check setting, model using, I/O dir ...



## Setting
my_dur = 2
DEMO = False #True
# model...2dcnn, 2dcrnn, 2dcrnn(stack cnn)...
# tb log dir
# output (e.g. submission) dir

train_csv_path='../input/train.csv'
train_audio_folder_path='../input/audio_train/'
test_csv_path='../input/sample_submission.csv'
test_audio_folder_path='../input/audio_test/'




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


## Models
def get_2d_conv_model(config):
    
    nclass = config.n_classes
    inp = Input(shape=(config.dim[0],config.dim[1],1))
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model



def operation2(x):
    xT = K.permute_dimensions(x,(0,2,1,3))  
    return xT

def get_2d_CRNN(config, time_fix=False):
    
    nclass = config.n_classes
    inp = Input(shape=(config.dim[0],config.dim[1],1))
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if time_fix: x = MaxPool2D(strides=(2,1),padding="same")(x)
    else: x = MaxPool2D()(x)

    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if time_fix: x = MaxPool2D(strides=(2,1),padding="same")(x)
    else: x = MaxPool2D()(x)
        
    x = Convolution2D(64, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)    
    if time_fix: x = MaxPool2D(strides=(2,1),padding="same")(x)
    else: x = MaxPool2D()(x)
    
    x = Lambda(operation2)(x)
    if time_fix: x = Reshape((int(x.shape[1]), int(x.shape[2]*x.shape[3])))(x)
    else: x = Reshape((int(x.shape[1]), int(x.shape[2]*x.shape[3])))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.1)(x)
    
    #x = Flatten()(x)
    x = Dense(64)(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


def get_2d_stack_CRNN(config, time_fix=False):
    
    nclass = config.n_classes
    inp = Input(shape=(config.dim[0],config.dim[1],1))
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if time_fix: x = MaxPool2D(strides=(2,1),padding="same")(x)
    else: x = MaxPool2D()(x)

    x = Convolution2D(32, (4,10), padding="same")(x)
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if time_fix: x = MaxPool2D(strides=(2,1),padding="same")(x)
    else: x = MaxPool2D()(x)
        
    x = Lambda(operation2)(x)
    if time_fix: x = Reshape((int(x.shape[1]), int(x.shape[2]*x.shape[3])))(x)
    else: x = Reshape((int(x.shape[1]), int(x.shape[2]*x.shape[3])))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.1)(x)
    
    #x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.1)(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model



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

LABELS = list(train.label.unique())
print ('Some example of labels',LABELS[:5])
print ('Unique num of labels',len(LABELS))


## Importing 2darray (data)
X_train  = np.load('../input/X_train_dur' + str(my_dur) + '.npy')
X_test = np.load('../input/X_test_dur' + str(my_dur) + '.npy')
y_train = np.load('../input/y_train.npy')
print ('Shape of Train, Test and Y', X_train.shape, X_test.shape, y_train.shape) 



## Training
if DEMO:
	config = Config(sampling_rate=44100, audio_duration=my_dur,learning_rate=0.001, 
	                use_mfcc=True, n_mfcc=40, max_epochs=1, n_folds=2)
	#X_train = X_train[:1000]
	#X_test = X_test[:1000]
	#y_train = y_train[:1000]
	#test = test[:1000]
	#skf = StratifiedKFold(train.label_idx[:1000], n_folds=config.n_folds)
else:
	config = Config(sampling_rate=44100, audio_duration=my_dur,learning_rate=0.001, 
	                use_mfcc=True, n_mfcc=40, max_epochs=75, n_folds=10)
	#skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)



PREDICTION_FOLDER = "pred_2d"
if not os.path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)
if os.path.exists('logs/' + PREDICTION_FOLDER):
    shutil.rmtree('logs/' + PREDICTION_FOLDER)


from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(width_shift_range=0.5,
                         fill_mode='nearest')  # horizontal_flip=True)

print ('DEMO',DEMO,'Epoch',config.max_epochs,'Folds',config.n_folds,'Duration',my_dur)
#for i, (train_split, val_split) in enumerate(skf):
for i in range(5,10):
    K.clear_session()
    train_split, val_split = np.load('skf/train_%d.npy'%i), np.load('skf/val_%d.npy'%i)
    print ('Some example of train/val split:', train_split[:5], val_split[:5])

    X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]
    checkpoint = ModelCheckpoint('best_%d.h5'%i, monitor='val_acc', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_acc", mode="auto", patience=20)
    tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%i'%i, write_graph=True)
    callbacks_list = [checkpoint, early, tb]  # tb
    print("#"*50)
    print("Fold: ", i)
    #model = get_2d_conv_model(config)
    model = get_2d_CRNN(config,time_fix=False)
    #model = get_2d_stack_CRNN(config,time_fix=False)
    #model = models.load_model('best_%d.h5'%i)    

    #history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list, 
    #                    batch_size=64, epochs=config.max_epochs, verbose=1)
    history = model.fit_generator(aug.flow(X,y, batch_size=64), validation_data=(X_val, y_val),
                                  steps_per_epoch = len(X)//64, validation_steps = len(X_val)//64,
                                  use_multiprocessing=True, workers=6, max_queue_size=20,
                                  callbacks=callbacks_list, epochs=config.max_epochs)
    model.load_weights('best_%d.h5'%i)

    # Save train predictions
    #predictions = model.predict(X_train, batch_size=64, verbose=1)
    #np.save("../temp/train_predictions_%d.npy"%i, predictions)

    # Save test predictions
    print ('Prediction on Testing')
    predictions = model.predict(X_test, batch_size=64, verbose=1)
    np.save("../temp/test_predictions_%d.npy"%i, predictions)

    # Make a submission file
    print ('Make submission file (fold %d)'%i)
    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test['label'] = predicted_labels
    test[['label']].to_csv("../temp/predictions_%d.csv"%i)

print ('Folds Training Done')




## Emsembled results (of k folds)
if DEMO:
	pred_list = []
	for i in range(2):
	    pred_list.append(np.load("../temp/test_predictions_%d.npy"%i))
	prediction = np.ones_like(pred_list[0])
	for pred in pred_list:
	    prediction = prediction*pred
	prediction = prediction**(1./len(pred_list))

	# Make a submission file
	top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
	predicted_labels = [' '.join(list(x)) for x in top_3]
	test = pd.read_csv('../input/sample_submission.csv')
	test = test[:1000]
	test['label'] = predicted_labels
	test[['fname', 'label']].to_csv("../output/2d_conv_ensembled_demo.csv", index=False)
else:
	pred_list = []
	for i in range(config.n_folds):
	    pred_list.append(np.load("../temp/test_predictions_%d.npy"%i))
	prediction = np.ones_like(pred_list[0])
	for pred in pred_list:
	    prediction = prediction*pred
	prediction = prediction**(1./len(pred_list))

	# Make a submission file
	top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
	predicted_labels = [' '.join(list(x)) for x in top_3]
	test = pd.read_csv('../input/sample_submission.csv')
	test['label'] = predicted_labels
	test[['fname', 'label']].to_csv("../output/2dcrnn_dur4_retime_aug2_ensb.csv", index=False)



