import numpy as np
import pickle
import cv2

from math import sqrt

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def iid_divide(l, g):
    '''
    divide list l among g groups
    each group has either int(len(l)/g) or int(len(l)/g)+1 elements
    returns a list of groups
    '''
    num_elems = len(l)
    group_size = int(len(l)/g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i : group_size * (i + 1)])
    bi = group_size*num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist

def get_indices(image_size, partitions, partition_index):
    '''
    get row and column indices for dividing a matrix of an image
    into several partitions and selecting one of them
    '''

    partition_size = int(image_size / sqrt(partitions))
    row_indices = [
        (partition_index[0]-1)*partition_size,
        (partition_index[0]-1)*partition_size+partition_size
    ]
    column_indices = [
        (partition_index[1]-1)*partition_size,
        (partition_index[1]-1)*partition_size+partition_size
    ]

    return(row_indices, column_indices)

def drop_pixels(dataset, partitions=4, keep_index=[1, 1]):
    '''
    split the pixels of a set of images into several partitions
    drop all partitions but one
    '''

    new_dataset = []
    image_size = len(dataset[0])
    row_indices, column_indices = get_indices(
        image_size,
        partitions,
        keep_index
    )

    for sample in dataset:
        if image_size == 28:
            new_sample = np.zeros((image_size, image_size))
        else:
            new_sample = np.zeros((image_size, image_size, 3))
        partition = []

        for row in sample[row_indices[0]:row_indices[1]]:
            new_row = row[column_indices[0]:column_indices[1]]
            partition.append(new_row)

        index = 0
        for row in new_sample[row_indices[0]:row_indices[1]]:
            row[column_indices[0]:column_indices[1]] = partition[index]
            index += 1

        new_dataset.append(new_sample)

    return(np.array(new_dataset))

def drop_pixels_federated(users, partitions):
    '''
    divide the pixels of a federated set of images into several partitions
    drop all partitions but one, leaving a large portion of blank pixels
    '''

    # Create list of indices for the partition
    dimension = int(sqrt(partitions))
    indices = []
    for first_item in range(dimension):
        for second_item in range(dimension):
            indices.append([first_item+1, second_item+1])

    user_range = int(len(users)/len(indices))

    for count, index in enumerate(indices):
        user_list = [
            count*user_range,
            count*user_range+user_range
        ]

        for user in range(user_list[0], user_list[1]):

            # Convert vectors to matrices
            matrices = []
            for vector in users[str(user)]['x']:
                length = len(vector)
                if length == 784:
                    matrix = np.array(vector).reshape([28, 28])
                else:
                    matrix = np.array(vector).reshape([32, 32, 3])
                matrices.append(matrix)

            # Drop some pixels
            matrices = drop_pixels(
                matrices,
                partitions=partitions,
                keep_index=index
            )

            # Convert matrices back to vectors
            vectors = []
            for matrix in matrices:
                vector = matrix.flatten().tolist()
                vectors.append(vector)

            # Replace the existing data
            users[str(user)]['x'] = vectors

    return(users)

def rescale(image, size):
    """Scale the image and revert back to its original scale"""

    original_size = np.shape(image)

    scaled = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    rescaled = cv2.resize(scaled, original_size, interpolation=cv2.INTER_AREA)

    return(rescaled)