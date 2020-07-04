import numpy as np
import math
import random
import os 
import torch

def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):       
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)

def convert_msg_to_label(pad_msg, dict_msg):
    nrows, ncols = pad_msg.shape
    labels = list()
    for i in range(nrows):
        column = list(set(list(pad_msg[i, :])))
        label = np.zeros(len(dict_msg))
        for c in column:
            label[c] = 1
        labels.append(label)
    return np.array(labels)


def mini_batches(X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0, shuffled=True):
    m = Y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    if shuffled == True:
        permutation = list(np.random.permutation(m))    
        shuffled_X_added = X_added_code[permutation, :, :, :, :]
        shuffled_X_removed = X_removed_code[permutation, :, :, :, :]
        if len(Y.shape) == 1:
            shuffled_Y = Y[permutation]
        else:
            shuffled_Y = Y[permutation, :]
    else:
        shuffled_X_added = X_added_code
        shuffled_X_removed = X_removed_code
        shuffled_Y = Y    
    

    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):                
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)        
    return mini_batches

def mini_batches_PNExtended(X_ftr, X_msg, X_added_code, X_removed_code, Y, shuffled=False, mini_batch_size=64, seed=0):
    m = Y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    if shuffled == True:
        permutation = list(np.random.permutation(m))
        shuffled_X_ftr = X_ftr[permutation, :]
        shuffled_X_msg = X_msg[permutation, :]
        shuffled_X_added = X_added_code[permutation, :, :, :, :]
        shuffled_X_removed = X_removed_code[permutation, :, :, :, :]
    else:
        shuffled_X_ftr = X_ftr
        shuffled_X_msg = X_msg
        shuffled_X_added = X_added_code
        shuffled_X_removed = X_removed_code

    if shuffled == True:
        if len(Y.shape) == 1:
            shuffled_Y = Y[permutation]
        else:
            shuffled_Y = Y[permutation, :]
    else:        
        shuffled_Y = Y        

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_ftr = shuffled_X_ftr[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]        
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_ftr, mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_ftr = shuffled_X_ftr[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]        
        mini_batch = (mini_batch_X_ftr, mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

