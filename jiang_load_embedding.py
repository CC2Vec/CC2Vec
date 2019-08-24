from parameters import read_args_jiang
from ultis import load_file
from jiang_prepared import load_Jiang_code_data
import os
import torch
from jiang_padding import padding_commit_code, commit_msg_label
from ultis import mini_batches_topwords
import datetime
from jiang_patch_embedding import PatchEmbedding
from jiang_train import reshape_code
import numpy as np


def commit_embedding(path, batches, model, params, nepoch):
    model.load_state_dict(torch.load(path))
    embedding_vectors, cnt = list(), 0
    print(path)
    for batch in batches:
        pad_added_code, pad_removed_code, labels = batch
        if torch.cuda.is_available():
            pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).cuda(), torch.tensor(
                pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
        else:
            pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).long(), torch.tensor(
                pad_removed_code).long(), torch.tensor(labels).float()

        # predict = model.forward(pad_msg, pad_added_code, pad_removed_code)
        commits_vector = model.forward_commit_embeds(pad_added_code, pad_removed_code)

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
    np.savetxt(path_save + 'epoch_' + str(nepoch) + '.txt', embedding_vectors)


def collect_batches(commit_diff, commit_msg, params, padding_code_info):
    max_line, max_length = padding_code_info[0], padding_code_info[1]
    pad_removed_code, pad_added_code, dict_code = padding_commit_code(data=commit_diff, max_line=max_line,
                                                                      max_length=max_length)
    pad_removed_code, pad_added_code = reshape_code(data=pad_removed_code), reshape_code(data=pad_added_code)
    print('Dictionary of commit code has size: %i' % (len(dict_code)))
    print('Shape of removed code and added code:', pad_removed_code.shape, pad_added_code.shape)

    labels, dict_msg = commit_msg_label(data=commit_msg)
    batches = mini_batches_topwords(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels,
                                    mini_batch_size=params.batch_size)

    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    params.vocab_code = len(dict_code) + 1
    params.code_line = max_line
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PatchEmbedding(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    return batches, model


if __name__ == '__main__':
    input_option = read_args_jiang().parse_args()
    input_help = read_args_jiang().print_help()

    # loading the commit code
    ##################################################################################
    ##################################################################################
    path_train_diff = './data/jiang_ase_2017/train.26208.diff'
    data_train_diff = load_Jiang_code_data(pfile=path_train_diff)
    path_test_diff = './data/jiang_ase_2017/test.3000.diff'
    data_test_diff = load_Jiang_code_data(pfile=path_test_diff)
    data_diff = data_train_diff + data_test_diff
    padding_code_info = (15, 40)  # max_line = 15; max_length = 40
    ##################################################################################
    ##################################################################################

    # loading the commit msg
    ##################################################################################
    ##################################################################################
    path_train_msg = './data/jiang_ase_2017/train.26208.msg'
    data_train_msg = load_file(path_file=path_train_msg)
    path_test_msg = './data/jiang_ase_2017/test.3000.msg'
    data_test_msg = load_file(path_file=path_test_msg)
    data_msg = data_train_msg + data_test_msg

    input_option.filter_sizes = [int(k) for k in input_option.filter_sizes.split(',')]
    # batches, model = collect_batches(commit_diff=data_diff[:500], commit_msg=data_msg[:500], params=input_option,
    #                                  padding_code_info=padding_code_info)
    batches, model = collect_batches(commit_diff=data_diff, commit_msg=data_msg, params=input_option,
                                     padding_code_info=padding_code_info)
    input_option.datetime = '2019-08-11_20-12-03'
    input_option.start_epoch = 1
    input_option.end_epoch = 50
    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        path_model = './snapshot/' + input_option.datetime + '/epoch_' + str(epoch) + '.pt'
        commit_embedding(path=path_model, batches=batches, model=model, params=input_option, nepoch=epoch)
