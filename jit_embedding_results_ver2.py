import pickle
from parameters import read_args
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from bfp_embedding_results import evaluation_metrics
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


def jit_clf_results(embedding_ftr, manual_ftr, labels, algorithm):
    print('Algorithm results:', algorithm)

    x_train = np.concatenate((embedding_ftr, manual_ftr), axis=1)
    x_train = preprocessing.scale(x_train)
    y_train = labels

    clf = None
    if algorithm == 'lr':
        clf = LogisticRegression().fit(X=x_train, y=y_train)
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


def jit_clf_results_ver2(embedding_ftr, manual_ftr, labels=None, algorithm=None, kfold=5):
    print('Algorithm results:', algorithm)

    embedding = np.concatenate((embedding_ftr, manual_ftr), axis=1)
    embedding = preprocessing.scale(embedding)

    skf = StratifiedKFold(n_splits=kfold)
    print('Algorithm results:', algorithm)
    accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
    for train_index, test_index in skf.split(embedding, labels):
        x_train, y_train = embedding[train_index], labels[train_index]
        x_test, y_test = embedding[test_index], labels[test_index]
        clf = None
        if algorithm == 'lr':
            clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train, y=y_train)
        elif algorithm == 'svm':
            clf = SVC(gamma='auto', probability=True, kernel='linear', max_iter=100).fit(X=x_train, y=y_train)
        elif algorithm == 'nb':
            clf = GaussianNB().fit(X=x_train, y=y_train)
        elif algorithm == 'dt':
            clf = DecisionTreeClassifier().fit(X=x_train, y=y_train)
        y_pred = clf.predict_proba(x_test)[:, 1]
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        accs.append(acc)
        prcs.append(prc)
        rcs.append(rc)
        f1s.append(f1)
        aucs.append(auc_)
    print('------------------------------------------------------------------------')
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (
        np.array(accs).mean(), np.array(prcs).mean(), np.array(rcs).mean(), np.array(f1s).mean(),
        np.array(aucs).mean()))


if __name__ == '__main__':
    path_data = './data/jit_openstack.pkl'
    # path_data = './data/jit_qt.pkl'
    with open(path_data, 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code, _ = data
    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    print('Shape of the commit message:', pad_msg.shape)
    print('Shape of the added/removed code:', (pad_added_code.shape, pad_removed_code.shape))
    print('Shape of the label of bug fixing patches:', labels.shape)
    print('Total words in the message dictionary: ', len(dict_msg))
    print('Total words in the code dictionary: ', len(dict_code))

    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    input_option.datetime = '2019-08-03_14-40-22'
    input_option.start_epoch = 10
    input_option.end_epoch = 10

    # input_option.datetime = '2019-08-03_14-45-06'
    # input_option.start_epoch = 10
    # input_option.end_epoch = 10

    path_raw_features = './data/jit_openstack_raw_features_.pkl'
    # path_raw_features = './data/jit_qt_raw_features_.pkl'
    with open(path_raw_features, 'rb') as input:
        data = pickle.load(input)
    indexes, raw_ftr, labels_rf = data

    algorithm, kfold = 'lr', 5
    # algorithm, kfold = 'dt', 5

    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        path_embedding = './embedding/' + input_option.datetime + '/epoch_' + str(epoch) + '.txt'
        embedding_ftr = np.loadtxt(path_embedding)  # be careful with the shape since we don't include the last batch
        embedding_ftr = embedding_ftr[indexes]
        jit_clf_results(embedding_ftr=embedding_ftr, manual_ftr=raw_ftr, labels=labels_rf, algorithm=algorithm)
        # jit_clf_results_ver2(embedding_ftr=embedding_ftr, manual_ftr=raw_ftr, labels=labels_rf, algorithm=algorithm,
        #                      kfold=5)
