# Cifar-10 Dataset

## Setup Instructions
- pip3 install numpy
- pip3 install pillow
- Run ```./preprocess.sh``` with a choice of the following tags:
    - ```-s``` := 'iid' to sample in an i.i.d. manner, or 'niid' to sample in a non-i.i.d. manner; more information on i.i.d. versus non-i.i.d. is included in the 'Notes' section
    - ```--nu``` := number of users; default is 90
    - ```--ns``` := number of training samples per class; default is 5400
    - ```--nt``` := number of test samples per class; default is 810
    - ```--np``` := number of image partitions; if it is more than 1 then the images will be divided into several partitions and only one of them is used to train the model; default is 0

i.e.
- ```./preprocess.sh -s iid --np 4 --nu 4 --ns 4800 --nt 720``` (i.i.d. dataset)<br/>
- ```./preprocess.sh -s niid --np 4 --nu 4 --ns 4800 --nt 720``` (non-i.i.d. dataset)

Make sure to delete the test and train subfolders in the data directory before re-running preprocess.sh

## Notes
- More details on i.i.d. versus non-i.i.d.:
  - In the i.i.d. sampling scenario, all users have examples of all classes.
  - In the non-i.i.d. sampling scenario, all users only have examples of two classes.
- Each .json file is an object with 3 keys:
  1. 'users', a list of users
  2. 'num_samples', a list of the number of samples for each user, and 
  3. 'user_data', an object with user names as keys and their respective data as values; for each user, data is represented as a list of images, with each image represented as a size-1024 integer list (flattened from 32 by 32)
- Run ```./stats.sh``` to get statistics of training data (JSON file in data/all_data/train must have been generated already)