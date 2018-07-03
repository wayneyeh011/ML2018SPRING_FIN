import os
import numpy as np
np.random.seed(1001)
import pandas as pd
from sklearn.cross_validation import StratifiedKFold


## NOTE
#  (1) Saving stratified folds in a dir called skf
#  (2) Training by loading these folds (index of val/train)



## SETTING
my_dur = 4

train_csv_path='../input/train.csv'
train_audio_folder_path='../input/audio_train/'
test_csv_path='../input/sample_submission.csv'
test_audio_folder_path='../input/audio_test/'



##
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


## 
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



##
config = Config(sampling_rate=44100, audio_duration=my_dur,learning_rate=0.001, 
	                use_mfcc=True, n_mfcc=40, max_epochs=70, n_folds=10)
skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)



##
if not os.path.exists('skf'):
    os.mkdir('skf')
    print ('making dir for index of eacf fold')
    
for i,(train_split, val_split) in enumerate(skf):
    print (train_split[:5], val_split[:5], len(train_split), len(val_split))
    np.save('skf/train_%d'%i,train_split)
    np.save('skf/val_%d'%i,val_split)















