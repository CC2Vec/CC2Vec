from parameters import read_args
import pickle
import numpy as np
from utilities import load_file
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utilities import convert_to_binary


def evaluation_metrics(path, labels):
    pred_score = load_file(path_file=path)
    pred_score = np.array([float(score) for score in pred_score])
    labels = labels[:pred_score.shape[0]]

    acc = accuracy_score(y_true=labels, y_pred=convert_to_binary(pred_score))
    prc = precision_score(y_true=labels, y_pred=convert_to_binary(pred_score))
    rc = recall_score(y_true=labels, y_pred=convert_to_binary(pred_score))
    f1 = f1_score(y_true=labels, y_pred=convert_to_binary(pred_score))
    auc = roc_auc_score(y_true=labels, y_score=pred_score)

    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc))


if __name__ == '__main__':

    # path_data = './data/jit_openstack.pkl'
    path_data = './data/jit_qt.pkl'
    with open(path_data, 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code, _ = data
    ##########################################################################################################
    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    print('Shape of the commit message:', pad_msg.shape)
    print('Shape of the added/removed code:', (pad_added_code.shape, pad_removed_code.shape))
    print('Shape of the label of bug fixing patches:', labels.shape)
    print('Total words in the message dictionary: ', len(dict_msg))
    print('Total words in the code dictionary: ', len(dict_code))

    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    # input_option.datetime = '2019-08-04_14-47-55'
    # input_option.start_epoch = 14
    # input_option.end_epoch = 14

    input_option.datetime = '2019-08-03_14-40-22'
    input_option.start_epoch = 10
    input_option.end_epoch = 10

    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        path_model = './results/' + input_option.datetime + '/epoch_' + str(epoch) + '.txt'
        print(path_model)
        evaluation_metrics(path=path_model, labels=labels)
