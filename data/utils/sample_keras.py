'''
samples from keras datasets;
by default samples in a non-iid manner, where each user only has samples
of two digits of image; otherwise, each user has samples from all digits
'''
import tensorflow as tf
import pandas as pd
import numpy as np


import argparse
import json
import os

from util import get_indices, drop_pixels, drop_pixels_federated, rescale

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
parser.add_argument('--nu',
                help="number of users; default: 90",
                type=int,
                default=90)
parser.add_argument('--ns',
                help="number of training samples; default: 5400",
                type=int,
                default=5400)
parser.add_argument('--nt',
                help="number of test samples; default: 810",
                type=int,
                default=810)
parser.add_argument('--np',
                help="split the images into several partitions",
                type=int,
                default=0)
parser.add_argument('--dx',
                help="horizontal size of rescaled images",
                type=int,
                default=28)
parser.add_argument('--dy',
                help="vertical size of rescaled images",
                type=int,
                default=28)
parser.set_defaults(iid=False)

args = parser.parse_args()

# Configuration
number_of_class = 10
if(args.iid):
    digit_per_user = 10
else:
    digit_per_user = 2

print('------------------------------')
print('sampling data')

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(parent_path, args.name, 'data')

# Load dataset
if args.name == 'fashion-mnist':
    fullset = tf.keras.datasets.fashion_mnist
elif args.name == 'cifar10':
    fullset = tf.keras.datasets.cifar10
else:
    fullset = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = fullset.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

if args.name == 'cifar10':
    y_train = y_train.flatten()
    y_test = y_test.flatten()

dataset = [
    [ x_train, y_train, args.ns, "train" ],
    [ x_test, y_test, args.nt, "test" ]
]

for count, subset in enumerate(dataset):

    # Rescale the training data
    if args.dx != 28 or args.dy != 28:
        if count == 0:
            new_set = []
            for record in subset[0]:
                record = rescale(record, (args.dx, args.dy))
                new_set.append(record)
            subset[0] = np.array(new_set)

    # Rename the variable
    y_list = subset[1]

    # Import to data frame
    data_frame = {'y': y_list}
    data_frame = pd.DataFrame(data=data_frame)

    # Sort the data
    data_frame = data_frame.sort_values(by=['y'])
    data_frame = data_frame.reset_index()
    data_frame.columns = ['x', 'y']

    # Balance the class distribution
    x_list = []
    y_list = []
    for digit in range(number_of_class):
        indices = data_frame[data_frame['y']==digit].x.tolist()[:subset[2]]
        x_list = x_list + subset[0][indices].tolist()
        y_list = y_list + subset[1][indices].tolist()

    # Flatten the features
    new_x_list = []
    for record in x_list:
        new_record = np.array(record).flatten().tolist()
        new_x_list.append(new_record)
    x_list = new_x_list

    # Split the data into multiple partitions
    if args.np != 0 and count == 0:
        x_list = x_list * args.np
        y_list = y_list * args.np

    # Assign user ID to each samples
    total_samples = len(x_list)
    if args.np != 0 and count == 0:
        replication = int(total_samples/args.np)
    else:
        replication = int(total_samples/args.nu/digit_per_user)
    counter = 0
    indices = []
    for user in range(args.nu):
        user_id = [user] * replication
        indices = indices + user_id
    if args.np != 0 and count == 0:
        pass
    else:
        indices = indices * digit_per_user

    # Import to data frame
    data_frame = {'x': x_list, 'y': y_list, 'id': indices}
    data_frame = pd.DataFrame(data=data_frame)

    # Initialize user data
    users = [str(i) for i in range(args.nu)]
    user_data = {}
    for user in users:
        user_data[user] = {'x': [], 'y': []}

    # Distribute the data
    for user in users:
        user_data[user]['x'] = data_frame[data_frame['id']==int(user)].x.tolist()
        user_data[user]['y'] = data_frame[data_frame['id']==int(user)].y.tolist()

    # Remove some pixels
    if args.np != 0 and count == 0:
        user_data = drop_pixels_federated(user_data, partitions=args.np)

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

    file_name = args.name + '_data_%s_%s.json' % (slabel, subset[3])
    output_file = os.path.join(data_dir, subset[3], file_name)

    print('writing %s' % file_name)
    with open(output_file, 'w') as outfile:
            json.dump(all_data, outfile)