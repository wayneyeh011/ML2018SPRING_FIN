import numpy as np 
import pandas as pd
import librosa
from keras.models import load_model
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D,
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from keras.utils import Sequence, to_categorical
import keras.backend as K
import sys

train_csv_path=sys.argv[1]
test_csv_path=sys.argv[2]

#---------------------------------1D_CNN--------------------------------------------------

print ('\n','Use 1D_cnn model')
COMPLETE_RUN = True
BATCH_SIZE=100 #必須為資料筆數的公因數(教學的datagenerator可能沒寫好)
MULTIPROCESSING=True #for windows : False
N_FOLDS=10

def get_1d_conv_model():

    nclass = 41
    input_length = 2*16000

    inp = Input(shape=(input_length,1))
    x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
    x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
    x = MaxPool1D(16)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.2)(x)

    x = Dense(64, activation=relu)(x)
    x = Dense(1028, activation=relu)(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


test = pd.read_csv(test_csv_path)
test.set_index("fname", inplace=True)
test_data=np.load('model/test_1d.npy')
for i  in range(10):
    model = get_1d_conv_model()
    model.load_weights('model/best_%d.h5'%i)
    locals()["d1_predictions_%s"%i] = model.predict(test_data, verbose=1 , batch_size=200)
#產出變數d1_predictions_0到d1_predictions_9
y_hat1d=np.ones((9400,41))
for i in range(10):
    y_hat1d=y_hat1d*locals()["d1_predictions_%s"%i]
y_hat1d=y_hat1d**(0.1)
np.save('y_hat1d.npy',y_hat1d)


#--------------------------------------CRNN(AUG, d2)----------------------------------------------------

print ('\n','Use CRNN(aug,dur=2) model')
X_test_path='model/X_test_dur2.npy'

train = pd.read_csv(train_csv_path)
test = pd.read_csv(test_csv_path)
LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])
LABELS = list(train.label.unique())
print ('Some example of labels',LABELS[:5])
print ('Unique num of labels',len(LABELS))
X_test = np.load(X_test_path)

y_hat_crnnd2=np.ones((9400,41))
for i in range(10):
    
    print("#"*50)
    print("Fold: ", i)
    model = models.load_model('model/crnn_d2a2_%d.h5'%i)

    # Save test predictions
    print ('Prediction on Testing')
    y_hat_crnnd2 =y_hat_crnnd2*model.predict(X_test, batch_size=64, verbose=1)
y_hat_crnnd2=y_hat_crnnd2**(0.1)
np.save('y_hat_crnnd2.npy',y_hat_crnnd2)


#--------------------------------------CRNN(AUG, d4)----------------------------------------------------

print ('\n','Use CRNN(aug,dur=4) model')
my_dur=4
ensb=False
X_test_path='model/X_test_dur4.npy'

train = pd.read_csv(train_csv_path)
test = pd.read_csv(test_csv_path)
LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])
LABELS = list(train.label.unique())
print ('Some example of labels',LABELS[:5])
print ('Unique num of labels',len(LABELS))
X_test = np.load(X_test_path)

y_hat_crnn=np.ones((9400,41))
for i in range(10):
    
    print("#"*50)
    print("Fold: ", i)
    model = models.load_model('model/crnn2_%d.h5'%i)

    # Save test predictions
    print ('Prediction on Testing')
    y_hat_crnn =y_hat_crnn*model.predict(X_test, batch_size=64, verbose=1)
y_hat_crnn=y_hat_crnn**(0.1)
np.save('y_hat_crnn.npy',y_hat_crnn)



#------------------------------------AugmentCNN(dur=2)-----------------------------------------------

print ('\n','Use 2DCNN(aug,dur=2) model')
X_test_path='model/X_test_dur2.npy'
train = pd.read_csv(train_csv_path)
test = pd.read_csv(test_csv_path)
LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])
LABELS = list(train.label.unique())
## Importing 2darray (data)
X_test = np.load(X_test_path)
y_hat_augd2=np.ones((9400,41))
for i in range(10):

    print("#"*50)
    print("Fold: ", i)
    model = models.load_model('model/cnn_d2a2_%d.h5'%i)

    # Save test predictions
    print ('Prediction on Testing')
    y_hat_augd2 = y_hat_augd2*model.predict(X_test, batch_size=64, verbose=1)
    
y_hat_augd2=y_hat_augd2**(1/10)
np.save('y_hat_augd2.npy',y_hat_augd2)


#------------------------------------AugmentCNN(dur=4)-----------------------------------------------

print ('\n','Use 2DCNN(aug,dur=4) model')
X_test_path='model/X_test_dur4.npy'
train = pd.read_csv(train_csv_path)
test = pd.read_csv(test_csv_path)
LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])
LABELS = list(train.label.unique())
## Importing 2darray (data)
X_test = np.load(X_test_path)
y_hat_augd4=np.ones((9400,41))
for i in range(10):

    print("#"*50)
    print("Fold: ", i)
    model = models.load_model('model/cnn_d4a2_%d.h5'%i)

    # Save test predictions
    print ('Prediction on Testing')
    y_hat_augd4 = y_hat_augd4*model.predict(X_test, batch_size=64, verbose=1)
    
y_hat_augd4=y_hat_augd4**(1/10)
np.save('y_hat_augd4.npy',y_hat_augd4)


#----------------------------------2D_CNN---------------------------------------------

print ('\n','Use 2DCNN(baseline) model')
train_label = pd.read_csv(train_csv_path)
encoded1=list(train_label.label.unique())
encoded=pd.get_dummies(train_label['label'])
encoded=list(encoded.columns)
sample_submit=pd.read_csv(test_csv_path)
id=sample_submit['fname'].tolist()
a=np.load("model/2D_data_pre.npy")

predict={}
for i in range(10):
    model=load_model('model/kfold%i.h5'%i)
    to_dataframe=model.predict(a,verbose=1)
    to_dataframe=pd.DataFrame(to_dataframe,columns=encoded)
    to_dataframe=to_dataframe[encoded1]
    predict["%i"%i]=np.array(to_dataframe)
y_hat2d=np.ones((9400,41))
for i in range(10):
    y_hat2d=y_hat2d*predict["%i"%i]
y_hat2d=y_hat2d**(0.1)
np.save('y_hat2d.npy',y_hat2d)


#--------------------------------------XGBOOST---------------------------------------

print ('\n','Use XGB model')
train = pd.read_csv(train_csv_path)
#給label數值
labels = list(train.label.unique())
label_to_idx=dict()
idx_to_label=dict()
idx=0
for label in labels:
    label_to_idx[label]=idx
    idx=idx+1
for key in label_to_idx:
    idx_to_label[label_to_idx[key]]=key


import numpy as np
test_276=np.load('model/xgb_processed_test_276.npy')
test_192=np.load('model/xgb_processed_test_192.npy')
test_132=np.load('model/xgb_processed_test_132.npy')


import numpy as np
test_audio_files = np.load('model/test_list.npy')

out_dict=dict()
num=0
for item in test_audio_files:
    out_dict[item]=num
    num=num+1

import pickle
clf_132 = pickle.load(open("model/xgb_132.pickle.dat", "rb"))
clf_192= pickle.load(open("model/xgb_192.pickle.dat", "rb"))
clf_276= pickle.load(open("model/xgb_276.pickle.dat", "rb"))


predict_132=clf_132.predict_proba(test_132)
predict_192=clf_192.predict_proba(test_192)
predict_276=clf_276.predict_proba(test_276)
output = pd.read_csv(test_csv_path)


predict_132_final=[]
predict_192_final=[]
predict_276_final=[]
for i in range(len(output)):
    name=output['fname'][i]
    predict_132_final.append(predict_132[out_dict[name]])
    predict_192_final.append(predict_192[out_dict[name]])
    predict_276_final.append(predict_276[out_dict[name]])
predict_132_final=np.array(predict_132_final)
predict_192_final=np.array(predict_192_final)
predict_276_final=np.array(predict_276_final)
y_hatxgboost=predict_132_final*predict_192_final*predict_276_final
y_hatxgboost=y_hatxgboost**(1/3)
np.save('y_hatxgboost.npy', y_hatxgboost)


#------------------------------------ensemble------------------------------------------

y_hat1d = np.load('y_hat1d.npy')
y_hat2d = np.load('y_hat2d.npy')
y_hat_augd2 = np.load('y_hat_augd2.npy')
y_hat_augd4 = np.load('y_hat_augd4.npy')
y_hatxgboost = np.load('y_hatxgboost.npy')
y_hat_crnnd2 = np.load('y_hat_crnnd2.npy')
y_hat_crnn = np.load('y_hat_crnn.npy')

print ('\n','Final ensembling')
y_hat=(y_hat1d**0.25)*(y_hat_augd2**0.15)*(y_hat_augd4**0.15)*(y_hatxgboost**0.2)*(y_hat_crnn**0.125)*(y_hat_crnnd2**0.125)
#y_hat=(y_hat1d**0.2)*(y_hat2d**0.)*(y_hat_augd2**0.)*(y_hat_augd4**0.)*(y_hatxgboost**)*(y_hat_crnn**)
#y_hat=(y_hat1d**0.2)*(y_hat2d**0.13)*(y_hat_augd2**0.13)*(y_hat_augd4**0.14)*(y_hatxgboost**0.2)*(y_hat_crnn**0.2)
#y_hat=(y_hat1d**0.20)*(y_hat2d**0.20)*(y_hatxgboost**0.175)*(y_hat_aug**0.225)*(y_hat_crnn**0.2)
#y_hat=0.175*y_hat1d+0.175*y_hat2d+0.075*y_hatxgboost+0.4*y_hat_aug+0.175*y_hat_crnn
y_hat=np.argsort(y_hat, axis=1)

for i in range(9400):
    predicted_labels=[' '.join([encoded1[y_hat[i,-1]],encoded1[y_hat[i,-2]],encoded1[y_hat[i,-3]]])]
    sample_submit['label'][i]=predicted_labels[0]
sample_submit.to_csv(sys.argv[3],index=False)

