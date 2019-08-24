from parameters import read_args_cnn
import pickle
from bfp_train import convert_msg_to_label
from bfp_train import mini_batches
import torch
from hierarchical_cnn import HierarchicalCNN
import numpy as np
import os


def load_model(data, params):
    pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    batches = mini_batches(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=pad_msg_labels,
                           mini_batch_size=input_option.batch_size)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.vocab_code = len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierarchicalCNN(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    return batches, model


def load_embedding(path, batches, model, params, nepoch):
    model.load_state_dict(torch.load(path))
    embedding_vectors, cnt = list(), 0
    with torch.no_grad():
        model.eval()
        for batch in batches:
            pad_added_code, pad_removed_code, labels = batch
            pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).cuda(), torch.tensor(
                pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)

            commits_vector = model.forward_commit_embeds_diff(pad_added_code, pad_removed_code)

            if torch.cuda.is_available():
                commits_vector = commits_vector.cpu().detach().numpy()
            else:
                commits_vector = commits_vector.detach().numpy()

            if cnt == 0:
                embedding_vectors = commits_vector
            else:
                embedding_vectors = np.concatenate((embedding_vectors, commits_vector), axis=0)
            print('Batch numbers:', cnt)
            cnt += 1
        path_save = './embedding/' + params.datetime + '/'
        save_folder = os.path.dirname(path_save)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        print(embedding_vectors.shape)
        np.savetxt(path_save + 'epoch_' + str(nepoch) + '.txt', embedding_vectors)


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

    with open('./data/linux_bfp.pickle', 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    ##########################################################################################################
    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    print('Shape of the commit message:', pad_msg.shape)
    print('Shape of the added/removed code:', (pad_added_code.shape, pad_removed_code.shape))
    print('Total words in the message dictionary: ', len(dict_msg))
    print('Total words in the code dictionary: ', len(dict_code))

    pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)
    print('Shape of the output labels: ', pad_msg_labels.shape)

    input_option = read_args_cnn().parse_args()
    input_help = read_args_cnn().print_help()

    input_option.datetime = '2019-07-21_20-56-53'
    input_option.start_epoch = 1
    input_option.end_epoch = 5

    data = (pad_added_code, pad_removed_code, pad_msg_labels, dict_msg, dict_code)
    batches, model = load_model(data=data, params=input_option)
    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        path_model = './snapshot/' + input_option.datetime + '/epoch_' + str(epoch) + '.pt'
        print(path_model)
        load_embedding(path=path_model, batches=batches, model=model, params=input_option, nepoch=epoch)
