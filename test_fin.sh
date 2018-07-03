## Download models (source Tien) (Warning: It's quite large)
wget -O model.tar.gz https://www.dropbox.com/s/4q8l9cr8syhrszz/model.tar.gz?dl=1
tar -xvzf model.tar.gz 
rm model.tar.gz
rm model/cnn* model/crnn*

## Download models (source Yeh)
wget -O d4_models.zip my_models.zip https://www.dropbox.com/sh/ccsvflnljg9awly/AADhqMrCUTweKx2Gpt8f3WIra?dl=1
unzip d4_models.zip
rm d4_models.zip X_test_dur4.npy 
wget -O d2_models.zip https://www.dropbox.com/sh/lbypf7ov0ui972o/AAA5Ku5t43bXa3fBP9hqqD4za?dl=1
unzip d2_models.zip
rm d2_models.zip X_test_dur2.npy
mv *h5 model/

## Execute python, argv: <train.csv> <test(submission).csv> <pred.csv>
python3 test_fin.py $1 $2 $3
