'''
samples from MNIST data;
by default samples in a non-iid manner, where each user only has samples
of two digits of image; otherwise, each user has samples from all digits
'''
import tensorflow as tf
import pandas as pd
import numpy as np


import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                help='name of dataset to parse; default: mnist;',
                type=str,
                default='mnist')
parser.add_argument('--iid',
                help='sample iid;',
                action="store_true")
parser.add_argument('--niid',
                help="sample niid;",
                dest='iid', action='store_false')
parser.set_defaults(iid=False)

args = parser.parse_args()

# Configuration
number_of_class = 10
train_sample_per_digit = 5400
test_sample_per_digit = 810
number_of_users = 90
if(args.iid):
    digit_per_user = 10
else:
    digit_per_user = 2

print('------------------------------')
print('sampling data')

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(parent_path, args.name, 'data')

# Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

dataset =   [
                    [ x_train, y_train, train_sample_per_digit, "train" ], 
			        [ x_test, y_test, test_sample_per_digit, "test" ]
				]

for subset in dataset:

    # Flatten the features
    x_list = []
    for record in subset[0]:
        new_record = record.flatten().tolist()
        x_list.append(new_record)

    # Rename the variable
    y_list = subset[1]

    # Import to data frame
    trainData = {'x': x_list, 'y': y_list}
    trainData = pd.DataFrame(data=trainData)

    # Sort the data
    trainData = trainData.sort_values(by=['y'])
    trainData = trainData.reset_index()

    # Balance the class distribution
    x_list = []
    y_list = []
    for digit in range(number_of_class):
        x = trainData[trainData['y']==digit].x.tolist()[:subset[2]]
        y = trainData[trainData['y']==digit].y.tolist()[:subset[2]]
        x_list = x_list + x
        y_list = y_list + y

    # Assign user ID to each samples
    total_samples = len(x_list)
    replication = int(total_samples/number_of_users/digit_per_user)
    counter = 0
    indices = []
    for user in range(number_of_users):
        user_id = [user] * replication
        indices = indices + user_id
    indices = indices * digit_per_user

    # Import to data frame
    trainData = {'x': x_list, 'y': y_list, 'id': indices}
    trainData = pd.DataFrame(data=trainData)

    # Initialize user data
    users = [str(i) for i in range(number_of_users)]
    user_data = {}
    for user in users:
        user_data[user] = {'x': [], 'y': []}

    # Distribute the data
    for user in users:
        user_data[user]['x'] = trainData[trainData['id']==int(user)].x.tolist()
        user_data[user]['y'] = trainData[trainData['id']==int(user)].y.tolist()

    # Prepare the content
    all_data = {}
    samples_per_user = [len(user_data[user]['y']) for user in users]
    all_data['users'] = users
    all_data['num_samples'] = samples_per_user
    all_data['user_data'] = user_data

    # Export to json file
    if(args.iid):
        slabel = 'iid'
    else:
       slabel = 'niid'

    file_name = 'mnist_data_%s_%s.json' % (slabel, subset[3])
    output_file = os.path.join(data_dir, subset[3], file_name)

    print('writing %s' % file_name)
    with open(output_file, 'w') as outfile:
            json.dump(all_data, outfile)