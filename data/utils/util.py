import pickle

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

def filter_list(data, patterns, exclude=False):
    '''
    data is list of original columns
    patterns is list of patterns, which indicate columns that will be excluded
    (if exclude flag is true) or included (if exclude flag is false)
    '''
    if type(patterns) != list:
        patterns = [patterns]

    for pattern in patterns:
        if exclude:
            data = [item for item in data if pattern not in item]
        else:
            data = [item for item in data if pattern in item]

    return(data)

def filter_time(data, start_date=None, end_date=None, time_column='Timestamp'):
    '''
    data is list of original columns
    patterns is list of patterns, which indicate columns that will be
    excluded (if exclude flag is true) or included (if exclude flag
    is false)
    '''

    column_index = data.columns.values.tolist().index(time_column)

    if start_date is None:
        start_date = min(data.iloc[:,column_index])
    if end_date is None:
        end_date = max(data.iloc[:,column_index])

    index = (
        data.iloc[:,column_index] >= start_date
    ) & (
        data.iloc[:,column_index] <= end_date
    )
    filtered_data = data.loc[index]
    filtered_data = filtered_data.reset_index(drop=True)

    return(filtered_data)