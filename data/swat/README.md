# SWaT Dataset

## Setup Instructions
- pip3 install numpy
- Run ```./preprocess.sh``` with a choice of the following tags:
    - ```--iu``` := number of users; default is 20
    - ```-k``` := number of samples per user; default is 7200
    - ```-y``` := target variable; default is ```LIT101```

i.e.
- ```./preprocess.sh --iu 20 -k 10800 -y LIT101``` (full-sized dataset)<br/>
- ```./preprocess.sh --iu 10 -k 3600 -y LIT101``` (small-sized dataset)

Make sure to have the raw data stored in ```data/raw_data``` directory before running preprocess.sh. The raw data can be obtained from [here](https://itrust.sutd.edu.sg/) 

## Notes
- More details on ```preprocess.sh```:
  - The order in which ```preprocess.sh``` processes data is 1. filter data based on timestamp and variables, 2. normalization, 3. creating train-test split.
  - Only data from 23 to 25 December 2015 is used for training data.
- Each .json file is an object with 3 keys:
  1. 'users', a list of users
  2. 'num_samples', a list of the number of samples for each user, and 
  3. 'user_data', an object with user names as keys and their respective data as values.
- Run ```./stats.sh``` to get statistics of training data (data in ```data/train/``` must have been generated already)
