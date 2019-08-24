import pickle
from parameters import read_args_cnn
import numpy as np
from utilities import mini_batches_extended_update
import torch
import os
import datetime
from hierarchical_cnn_jit_classification import DeepJITExtented
import torch.nn as nn
from parameters import print_params


def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)


def running_train(batches, model, params):
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps, num_epoch = 0, 1
    for epoch in range(1, params.num_epochs + 1):
        for batch in batches:
            pad_ftr, pad_msg, pad_added_code, pad_removed_code, labels = batch
            pad_ftr, pad_msg, pad_added_code, pad_removed_code, labels = torch.cuda.DoubleTensor(pad_ftr), torch.tensor(
                pad_msg).cuda(), torch.tensor(pad_added_code).cuda(), torch.tensor(
                pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)

            optimizer.zero_grad()
            predict = model.forward(pad_ftr, pad_msg, pad_added_code, pad_removed_code)
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
    pad_extended_ftr, pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data

    nrows = pad_extended_ftr.shape[0]
    pad_msg = pad_msg[:nrows, :]
    pad_added_code = pad_added_code[:nrows, :, :, :]
    pad_removed_code = pad_removed_code[:nrows, :, :, :]
    labels = labels[:nrows]

    batches = mini_batches_extended_update(X_ftr=pad_extended_ftr, X_msg=pad_msg, X_added_code=pad_added_code,
                                           X_removed_code=pad_removed_code, Y=labels,
                                           mini_batch_size=input_option.batch_size)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    params.n_file = pad_added_code.shape[1]
    params.n_line = pad_added_code.shape[2]
    params.n_token = pad_added_code.shape[3]
    print_params(params=params)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepJITExtented(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
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

    # path_data = './data/jit_openstack.pkl'
    path_data = './data/jit_qt.pkl'
    with open(path_data, 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code, _ = data
    ##########################################################################################################
    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    print('Shape of the commit message:', pad_msg.shape)
    print('Shape of the added/removed code:', (pad_added_code.shape, pad_removed_code.shape))
    print('Total words in the message dictionary: ', len(dict_msg))
    print('Total words in the code dictionary: ', len(dict_code))

    input_option = read_args_cnn().parse_args()
    input_help = read_args_cnn().print_help()

    # input_option.datetime = '2019-08-03_14-40-22'
    # path_embedding = './embedding/' + input_option.datetime + '/epoch_10.txt'

    input_option.datetime = '2019-08-03_14-45-06'
    path_embedding = './embedding/' + input_option.datetime + '/epoch_10.txt'

    embedding_ftr = np.loadtxt(path_embedding)  # be careful with the shape since we don't include the last batch
    print('Shape of the extended features:', embedding_ftr.shape)

    data = (embedding_ftr, pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code)
    input_option.extended_ftr = embedding_ftr.shape[1]
    train_model(data=data, params=input_option)
