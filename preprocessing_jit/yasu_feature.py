import pandas as pd
import numpy as np
import pickle


def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    return df.values


def get_features(data):
    # return the features of yasu data
    return data[:, 4:33]


def get_ids(data):
    # return the labels of yasu data
    return data[:, 1:2].flatten().tolist()


def get_label(data):
    data = data[:, 3:4].flatten().tolist()
    data = [1 if int(d) > 0 else 0 for d in data]
    return data


def load_df_yasu_data(path_data):
    data = pd.read_csv(path_data)
    data = replace_value_dataframe(df=data)
    ids, labels, features = get_ids(data=data), get_label(data=data), get_features(data=data)
    indexes, new_ids, new_labels, new_features = list(), list(), list(), list()
    cnt_noexits = 0
    for i in range(0, len(ids)):
        try:
            indexes.append(i)
        except FileNotFoundError:
            print('File commit id no exits', ids[i], cnt_noexits)
            cnt_noexits += 1
    ids = [ids[i] for i in indexes]
    labels = [labels[i] for i in indexes]
    features = features[indexes]
    return (ids, np.array(labels), features)


def load_yasu_data(project):
    if project == 'openstack':
        path_data = '../data/jit_defect/yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.all.8.csv'
    elif project == 'qt':
        path_data = '../data/jit_defect/yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.all.10.csv'
    else:
        print('Please type the correct name of the project')
        exit()
    data = load_df_yasu_data(path_data=path_data)
    return data


if __name__ == '__main__':
    # project = 'openstack'
    project = 'qt'
    rf = load_yasu_data(project=project)

    # path_data = '../data/jit_openstack.pkl'
    path_data = '../data/jit_qt.pkl'
    with open(path_data, 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code, ids = data
    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    print('Shape of the commit message:', pad_msg.shape)
    print('Shape of the added/removed code:', (pad_added_code.shape, pad_removed_code.shape))
    print('Shape of the label of bug fixing patches:', labels.shape)
    print('Total words in the message dictionary: ', len(dict_msg))
    print('Total words in the code dictionary: ', len(dict_code))
    ids = ids[:int(len(ids) / 64) * 64]

    ids_rf, labels_rf, features_rf = rf
    interset_id = list(set(ids) & set(ids_rf))
    indexes = [ids.index(id) for id in interset_id]
    indexes_raw_ftr = [ids_rf.index(id) for id in interset_id]
    features_rf = features_rf[indexes_raw_ftr]
    labels_rf = labels_rf[indexes_raw_ftr]
    print(len(indexes), features_rf.shape)

    raw_ftr_data = (indexes, features_rf, labels_rf)
    write_data = open('../data/jit_' + project + '_raw_features_.pkl', 'wb')
    pickle.dump(raw_ftr_data, write_data)
