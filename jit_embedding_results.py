from parameters import read_args
import pickle
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score


def evaluation_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    return acc, prc, rc, f1, auc_


def bfp_clf_results(path, labels=None, algorithm=None, kfold=5):
    embedding = np.loadtxt(path)  # be careful with the shape since we don't include the last batch
    nrows = embedding.shape[0]
    labels = labels[:nrows]
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
    np.array(accs).mean(), np.array(prcs).mean(), np.array(rcs).mean(), np.array(f1s).mean(), np.array(aucs).mean()))


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

    # input_option.datetime = '2019-08-03_14-40-22'
    # input_option.start_epoch = 10
    # input_option.end_epoch = 10

    input_option.datetime = '2019-08-03_14-45-06'
    input_option.start_epoch = 10
    input_option.end_epoch = 10

    algorithm, kfold = 'lr', 2
    # algorithm, kfold = 'svm', 5
    # algorithm, kfold = 'nb', 1
    # algorithm, kfold = 'dt', 5

    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        path_model = './embedding/' + input_option.datetime + '/epoch_' + str(epoch) + '.txt'
        bfp_clf_results(path=path_model, labels=labels, algorithm=algorithm, kfold=kfold)
        # exit()
