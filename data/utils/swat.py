'''
samples from SWaT data;
'''
import pandas as pd
import numpy as np


import argparse
import json
import os

from util import filter_list, filter_time
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                help='name of dataset to parse; default: swat',
                type=str,
                default='swat')
parser.add_argument('--u',
                help=('number of users; default: 20;'),
                type=int,
                default=20)
parser.add_argument('--samples',
                help='number of samples per user; default: 7200;',
                type=int,
                default=7200)
parser.add_argument('--target',
                help='sensor or actuator in the system; default: LIT101;',
                type=str,
                default='LIT101')

args = parser.parse_args()

# Configuration
time_column = 'Timestamp'
label_column = 'Status'
train_sample_per_user = args.samples
test_sample_per_user = int(86400/args.u)

print('------------------------------')
print('sampling data')

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(parent_path, args.name, 'data')

# Load dataset
swat = pd.read_csv(data_dir + '/raw_data/swat_normal_dataset.csv')

# Filter based on time range 
swat[time_column] = pd.to_datetime(swat[time_column], errors='coerce')
swat[label_column] = swat[label_column].astype('category')
training = filter_time(
    swat, 
    start_date='2015-12-23 00:00:00',
    end_date='2015-12-25 23:59:59'
)
test = filter_time(
    swat, 
    start_date='2015-12-26 00:00:00',
    end_date='2015-12-26 23:59:59'
)

# Filter based on number of samples
training = training.iloc[:train_sample_per_user*args.u]
test = test.iloc[:test_sample_per_user*args.u]

# Define input features and target
all_columns = swat.columns.values.tolist()
unwanted_columns = [time_column, label_column, args.target]
features = filter_list(all_columns, unwanted_columns, exclude=True)

# Split input features and target
x_train = training.loc[:, features].values.tolist()
y_train = training.loc[:, args.target].values.tolist()
x_test = test.loc[:, features].values.tolist()
y_test = test.loc[:, args.target].values.tolist()

# Normalize the input features
scaler = StandardScaler()
scaler.fit(x_train)
training = pd.DataFrame(scaler.transform(x_train))
test = pd.DataFrame(scaler.transform(x_test))

# Join input features and the target
training.columns = features
test.columns = features
training[args.target] = y_train
test[args.target] = y_test

# Apply operations on training and test set
dataset = [
    [training, train_sample_per_user, 'train'], 
    [test, test_sample_per_user, 'test']
]
for subset in dataset:
    
    # Assign user ID to each samples
    indices = []
    for user in range(args.u):
        user_id = [user] * subset[1]
        indices = indices + user_id
    subset[0]['id'] = indices

    # Initialize user data
    users = [str(user_id) for user_id in range(args.u)]
    user_data = {}
    for user in users:
        user_data[user] = {'x': [], 'y': []}

    # Distribute the data
    for user in users:
        indices = subset[0][subset[0]['id']==int(user)].index
        user_data[user]['x'] = subset[0].loc[indices, features].values.tolist()
        user_data[user]['y'] = subset[0].loc[indices, args.target].values.tolist()

    # Prepare the content
    all_data = {}
    samples_per_user = [len(user_data[user]['y']) for user in users]
    all_data['users'] = users
    all_data['num_samples'] = samples_per_user
    all_data['user_data'] = user_data

    # Export to json file
    file_name = 'mnist_data_%s.json' % (subset[2])
    output_file = os.path.join(data_dir, subset[2], file_name)

    print('writing %s' % file_name)
    with open(output_file, 'w') as outfile:
            json.dump(all_data, outfile)