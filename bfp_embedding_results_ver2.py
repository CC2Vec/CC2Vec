import pickle
from parameters import read_args
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from bfp_embedding_results import evaluation_metrics


def bfp_clf_results(embedding_ftr, manual_ftr, labels, algorithm):
    print('Algorithm results:', algorithm)

    x_train = np.concatenate((embedding_ftr, manual_ftr), axis=1)
    y_train = labels

    clf = None
    if algorithm == 'lr':
        clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train, y=y_train)
    elif algorithm == 'svm':
        clf = SVC(gamma='auto', probability=True, kernel='linear', max_iter=100).fit(X=x_train, y=y_train)
    elif algorithm == 'nb':
        clf = GaussianNB().fit(X=x_train, y=y_train)
    elif algorithm == 'dt':
        clf = DecisionTreeClassifier().fit(X=x_train, y=y_train)
    y_pred = clf.predict_proba(x_train)[:, 1]
    acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_train, y_pred=y_pred)
    print('------------------------------------------------------------------------')
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))


if __name__ == '__main__':
    with open('./data/bfp_linux.pickle', 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    ##########################################################################################################
    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    print('Shape of the commit message:', pad_msg.shape)
    print('Shape of the added/removed code:', (pad_added_code.shape, pad_removed_code.shape))
    print('Shape of the label of bug fixing patches:', labels.shape)
    print('Total words in the message dictionary: ', len(dict_msg))
    print('Total words in the code dictionary: ', len(dict_code))

    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    input_option.datetime = '2019-07-21_20-56-53'
    input_option.start_epoch = 5
    input_option.end_epoch = 5

    with open('./data/bfp_linux_raw_features.pickle', 'rb') as input:
        data = pickle.load(input)
    indexes, raw_ftr = data
    raw_ftr = raw_ftr.iloc[:, 1:]
    raw_ftr = np.asarray(raw_ftr)

    algorithm, kfold = 'lr', 5
    # algorithm, kfold = 'svm', 5
    # algorithm, kfold = 'nb', 5
    # algorithm, kfold = 'dt', 5

    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        path_embedding = './embedding/' + input_option.datetime + '/epoch_' + str(epoch) + '.txt'
        embedding_ftr = np.loadtxt(path_embedding)  # be careful with the shape since we don't include the last batch
        embedding_ftr, labels = embedding_ftr[indexes], labels[indexes]
        bfp_clf_results(embedding_ftr=embedding_ftr, manual_ftr=raw_ftr, labels=labels, algorithm=algorithm)
