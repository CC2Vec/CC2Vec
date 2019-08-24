import os
import math
import numpy as np
import random


def load_file(path_file):
    lines = list(open(path_file, 'r', encoding='utf8', errors='ignore').readlines())
    lines = [l.strip() for l in lines]
    return lines


def print_label(data):
    print(data[0])


def get_label(data):
    labels = list()
    for i in range(1, len(data)):
        split_i = data[i].split(',')
        if split_i[2] == '':
            labels.append('False')
        else:
            if int(split_i[2]) > 0:
                labels.append('True')
            else:
                labels.append('False')
    return labels


def commit_id(data):
    ids = list()
    for i in range(1, len(data)):
        split_ = data[i].split(',')
        ids.append(split_[0])
    return ids


def dict_label(commit_ids, labels):
    dicts = list()
    for c, l in zip(commit_ids, labels):
        dictionary = dict()
        dictionary['id'] = c
        dictionary['label'] = l
        dicts.append(dictionary)
    return dicts


def write_file(path_file, data):
    split_path = path_file.split("/")
    path_ = split_path[:len(split_path) - 1]
    path_ = "/".join(path_)

    if not os.path.exists(path_):
        os.makedirs(path_)

    with open(path_file, 'w') as out_file:
        for line in data:
            # write line to output file
            out_file.write(str(line))
            out_file.write("\n")
        out_file.close()


def mini_batches(X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg, shuffled_X_code, shuffled_Y = X_msg, X_code, Y

    # Step 2: Partition (X, Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitioning
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_code = shuffled_X_code[num_complete_minibatches * mini_batch_size: m, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_update(X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg, shuffled_X_code, shuffled_Y = X_msg, X_code, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X_msg, mini_batch_X_code = shuffled_X_msg[indexes], shuffled_X_code[indexes]
        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_undersampling(X_msg, X_code, Y, mini_batch_size=64, seed=0):
    # Step 1: No shuffle (X, Y)
    shuffled_X_msg, shuffled_X_code, shuffled_Y = X_msg, X_code, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # Step 2: apply under sampling. Reduce the size of the majority class
    indexes = sorted(Y_pos + random.sample(Y_neg, len(Y_pos)))
    sample_shuffled_X_msg, sample_shuffled_X_code = shuffled_X_msg[indexes], shuffled_X_code[indexes]
    sample_shuffled_Y = shuffled_Y[indexes]
    batches = mini_batches(X_msg=sample_shuffled_X_msg, X_code=sample_shuffled_X_code, Y=sample_shuffled_Y,
                           mini_batch_size=mini_batch_size)
    return batches


if __name__ == '__main__':
    project = 'openstack'
    # project = 'qt'
    path_labels = './labels/' + project + '.csv'
    data_labels = load_file(path_file=path_labels)
    ids, labels = commit_id(data=data_labels), get_label(data=data_labels)
    # dict_label(commit_ids=ids, labels=labels)
    print(len(ids), len(labels))
    print_label(data=data_labels)

    valid_ids = load_file(path_file='./labels/' + project + '_ids.txt')
    print(len(valid_ids))
    data = list()
    for i, l in zip(ids, labels):
        if i in valid_ids:
            print(i, l)
            data.append(i + '\t' + l)
    write_file(path_file='./labels/' + project + '_ids_label.txt', data=data)