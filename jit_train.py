import pickle
from parameters import read_args
import numpy as np
from utilities import mini_batches
import torch
import os
from hierarchical_attention_jit import JIT_HierachicalRNN
import torch.nn as nn
import datetime
from parameters import print_params


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


def running_train(batches, model, params):
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps, num_epoch = 0, 1
    for epoch in range(1, params.num_epochs + 1):
        for batch in batches:
            # reset the hidden state of hierarchical attention model
            state_word = model.init_hidden_word()
            state_sent = model.init_hidden_sent()
            state_hunk = model.init_hidden_hunk()

            pad_added_code, pad_removed_code, labels = batch
            labels = torch.cuda.FloatTensor(labels)
            optimizer.zero_grad()
            predict = model.forward(pad_added_code, pad_removed_code, state_hunk, state_sent, state_word)
            loss = nn.BCELoss()
            loss = loss(predict, labels)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % params.log_interval == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(num_epoch, steps, loss.item()))

        save(model, params.save_dir, 'epoch', num_epoch)
        num_epoch += 1


def train_model(data, params):
    pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    batches = mini_batches(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=pad_msg_labels,
                           mini_batch_size=input_option.batch_size)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)
    print_params(params=params)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hierarchical_attention = JIT_HierachicalRNN(args=params)
    if torch.cuda.is_available():
        model = hierarchical_attention.cuda()
    running_train(batches=batches, model=model, params=params)


if __name__ == '__main__':
    # loading data
    ##########################################################################################################
    # data = pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code
    # pad_msg: padding commit message
    # pad_added_code: padding added code
    # pad_removed_code: padding removed code
    # labels: label of our data, stable or non-stable patches
    # dict_msg: dictionary of commit message
    # dict_code: dictionary of commit code

    path_data = './data/jit_openstack.pkl'
    # path_data = './data/jit_qt.pkl'
    with open(path_data, 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code, _ = data
    ##########################################################################################################
    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    print('Shape of the commit message:', pad_msg.shape)
    print('Shape of the added/removed code:', (pad_added_code.shape, pad_removed_code.shape))
    print('Total words in the message dictionary: ', len(dict_msg))
    print('Total words in the code dictionary: ', len(dict_code))

    pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)
    print('Shape of the output labels: ', pad_msg_labels.shape)

    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    input_option.embed_size = 64
    input_option.hidden_size = 32

    data = (pad_added_code, pad_removed_code, pad_msg_labels, dict_msg, dict_code)
    train_model(data=data, params=input_option)
