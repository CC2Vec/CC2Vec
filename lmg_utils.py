import os 
import torch
import numpy as np
import math

def commit_msg_label(data, dict_msg):
    labels_ = np.array([1 if w in d.split() else 0 for d in data for w in dict_msg])
    labels_ = np.reshape(labels_, (int(labels_.shape[0] / len(dict_msg)), len(dict_msg)))
    return labels_

def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):       
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)

def mini_batches(X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0, Shuffled=True):
    m = Y.shape[0]  # number of training examples
    mini_batches = []

    if Shuffled == True:
        np.random.seed(seed)
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))    
        shuffled_X_added = X_added_code[permutation, :, :]
        shuffled_X_removed = X_removed_code[permutation, :, :]
        
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
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)        
    return mini_batches