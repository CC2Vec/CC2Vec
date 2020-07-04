import numpy as np
import math
import os, torch
import random

def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):       
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)

def mini_batches(X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0, shuffled=True):
    m = Y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    if shuffled == True:
        permutation = list(np.random.permutation(m))    
        shuffled_X_added = X_added_code[permutation, :, :, :]
        shuffled_X_removed = X_removed_code[permutation, :, :, :]
        
        if len(Y.shape) == 1:
            shuffled_Y = Y[permutation]
        else:
            shuffled_Y = Y[permutation, :]
    else:
        shuffled_X_added = X_added_code
        shuffled_X_removed = X_removed_code
        shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):                
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)        
    return mini_batches

def mini_batches_DExtended(X_ftr, X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    shuffled_X_ftr, shuffled_X_msg, shuffled_X_code, shuffled_Y = X_ftr, X_msg, X_code, Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X_ftr = shuffled_X_ftr[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_ftr, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_ftr = shuffled_X_ftr[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_code = shuffled_X_code[num_complete_minibatches * mini_batch_size: m, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_ftr, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def mini_batches_update_DExtended(X_ftr, X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_ftr, shuffled_X_msg, shuffled_X_code, shuffled_Y = X_ftr, X_msg, X_code, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]    

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X_ftr = shuffled_X_ftr[indexes]
        mini_batch_X_msg, mini_batch_X_code = shuffled_X_msg[indexes], shuffled_X_code[indexes]
        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (mini_batch_X_ftr, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches