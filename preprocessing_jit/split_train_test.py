from padding import dictionary_commit, padding_message, padding_commit_code, mapping_dict_msg, mapping_dict_code
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, KFold
import pickle


def convert_label(data):
    return np.array([0 if d == 'False' else 1 for d in data])


def info_label(data):
    pos = [d for d in data if d == 1]
    neg = [d for d in data if d == 0]
    print('Positive: %i -- Negative: %i' % (len(pos), len(neg)))


def get_index(data, index):
    return [data[i] for i in index]


def clean_and_reformat_code(data):
    # remove empty lines in code
    # divide code to two part: added_code and removed_code
    new_diff_added_code, new_diff_removed_code = list(), list()
    for diff in data:
        files = list()
        for file in diff:
            lines = file['added_code']
            new_lines = list()
            for line in lines:
                if len(line.strip()) > 0:
                    new_lines.append(line)
            files.append(new_lines)
        new_diff_added_code.append(files)
    for diff in data:
        files = list()
        for file in diff:
            lines = file['removed_code']
            new_lines = list()
            for line in lines:
                if len(line.strip()) > 0:
                    new_lines.append(line)
            files.append(new_lines)
        new_diff_removed_code.append(files)
    return (new_diff_added_code, new_diff_removed_code)


def folding_data(pad_msg, pad_code, labels, ids, n_folds):
    sss = StratifiedShuffleSplit(n_splits=n_folds, random_state=0)  # random_state = 0 --default setting
    added_code, removed_code = pad_code
    for train_index, test_index in sss.split(pad_msg, labels):
        pad_msg_train, pad_msg_test = get_index(data=pad_msg, index=train_index), get_index(data=pad_msg,
                                                                                            index=test_index)
        added_code_train, added_code_test = get_index(data=added_code, index=train_index), get_index(data=added_code,
                                                                                                     index=test_index)
        removed_code_train, removed_code_test = get_index(data=removed_code, index=train_index), get_index(
            data=removed_code,
            index=test_index)
        labels_train, labels_test = labels[train_index], labels[test_index]
        ids_train, ids_test = get_index(data=ids, index=train_index), get_index(data=ids, index=test_index)
        train = (ids_train, labels_train, pad_msg_train, added_code_train, removed_code_train)
        test = (ids_test, labels_test, pad_msg_test, added_code_test, removed_code_test)
        return train, test


def folding_data_authordate(pad_msg, pad_code, labels, ids, n_folds):
    kf = KFold(n_splits=n_folds, random_state=0)
    indexes = list(kf.split(pad_msg))
    train_index, test_index = indexes[len(indexes) - 1]

    pad_msg_train, pad_msg_test = get_index(data=pad_msg, index=train_index), get_index(data=pad_msg,
                                                                                        index=test_index)
    pad_code_train, pad_code_test = get_index(data=pad_code, index=train_index), get_index(data=pad_code,
                                                                                           index=test_index)
    labels_train, labels_test = labels[train_index], labels[test_index]
    info_label(data=labels_train)
    info_label(data=labels_test)
    ids_train, ids_test = get_index(data=ids, index=train_index), get_index(data=ids, index=test_index)
    train = (ids_train, labels_train, pad_msg_train, pad_code_train)
    test = (ids_test, labels_test, pad_msg_test, pad_code_test)
    return train, test


if __name__ == '__main__':
    project = 'openstack'
    # project = 'qt'
    f = open('../data/jit_defect/' + project + '.pkl', 'rb')
    obj = pickle.load(f)
    f.close()

    messages, codes, labels, ids = obj
    labels = convert_label(labels)
    codes = clean_and_reformat_code(data=codes)

    info_label(data=labels)
    print('Number of instances in commit message %i and commit code %i ' % (len(messages), len(codes)))
    print('Labels: %i' % (len(labels)))

    dict_msg, dict_code = dictionary_commit(data=messages, type_data='msg'), dictionary_commit(data=codes,
                                                                                               type_data='code')
    pad_msg = padding_message(data=messages, max_length=256)
    added_code, removed_code = codes
    pad_added_code = padding_commit_code(data=added_code, max_file=3, max_line=10, max_length=256)
    pad_removed_code = padding_commit_code(data=removed_code, max_file=3, max_line=10, max_length=256)

    pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
    pad_added_code = mapping_dict_code(pad_code=pad_added_code, dict_code=dict_code)
    pad_removed_code = mapping_dict_code(pad_code=pad_removed_code, dict_code=dict_code)
    data = (pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code, ids)
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))
    print('Shape of commit message:', pad_msg.shape)
    print('Shape of added code:', pad_added_code.shape)
    print('Shape of removed code:', pad_removed_code.shape)
    print('Shape of labels:', labels.shape)
    print('Ids of projects:', project, len(ids))
    write_data = open('../data/jit_' + project + '.pkl', 'wb')
    pickle.dump(data, write_data)
