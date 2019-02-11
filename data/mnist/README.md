# MNIST Dataset

## Setup Instructions
- pip3 install numpy
- pip3 install pillow
- Run ```./preprocess.sh``` with a choice of the following tags:
    - ```-s``` := 'iid' to sample in an i.i.d. manner, or 'niid' to sample in a non-i.i.d. manner; more information on i.i.d. versus non-i.i.d. is included in the 'Notes' section

i.e.
- ```./preprocess.sh -s iid``` (i.i.d. dataset)<br/>
- ```./preprocess.sh -s niid``` (non-i.i.d. dataset)

Make sure to delete the test and train subfolders in the data directory before re-running preprocess.sh

## Notes
- More details on i.i.d. versus non-i.i.d.:
  - In the i.i.d. sampling scenario, all users have examples of all digits.
  - In the non-i.i.d. sampling scenario, all users only have examples of two digits.
- Each .json file is an object with 3 keys:
  1. 'users', a list of users
  2. 'num_samples', a list of the number of samples for each user, and 
  3. 'user_data', an object with user names as keys and their respective data as values; for each user, data is represented as a list of images, with each image represented as a size-784 integer list (flattened from 28 by 28)
- Run ```./stats.sh``` to get statistics of training data (JSON file in data/all_data/train must have been generated already)