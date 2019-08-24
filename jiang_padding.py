import itertools
import numpy as np
from ultis import load_file


def dictionary_code(data):
    # create dictionary for commit message
    tokens_removed = [line.split() for commit in data for line in commit['removed']]
    tokens_removed = list(itertools.chain.from_iterable(tokens_removed))

    tokens_added = [line.split() for commit in data for line in commit['added']]
    tokens_added = list(itertools.chain.from_iterable(tokens_added))

    tokens = list(set(tokens_removed + tokens_added))
    tokens.append('<NULL>')
    new_dict = dict()
    for i in range(len(tokens)):
        new_dict[tokens[i]] = i
    return new_dict


def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        return str(line + ' <NULL>' * (max_length - line_length)).strip()
    elif line_length > max_length:
        line_split = line.split()
        return " ".join([line_split[i] for i in range(max_length)])
    else:
        return line


def padding_multiple_length(lines, max_length):
    return [padding_length(line=l, max_length=max_length) for l in lines]


def padding_commit_code_length(data, max_length):
    return [padding_multiple_length(lines=commit, max_length=max_length) for commit in data]


def padding_commit_code_line(data, max_line, max_length):
    new_data = list()
    for d in data:
        if len(d) == max_line:
            new_data.append(d)
        elif len(d) > max_line:
            new_data.append(d[:max_line])
        else:
            num_added_line = max_line - len(d)
            for i in range(num_added_line):
                d.append(('<NULL> ' * max_length).strip())
            new_data.append(d)
    return new_data


def mapping_commit_code(data, max_line, max_length):
    padding_length = padding_commit_code_length(data=data, max_length=max_length)
    padding_line = padding_commit_code_line(padding_length, max_line=max_line, max_length=max_length)
    return padding_line


def mapping_commit_dict_code(data, dict_code):
    new_c = list()
    for c in data:
        new_l = list()
        for l in c:
            l_split, new_t = l.split(), list()
            for t in l_split:
                new_t.append(dict_code[t])
            new_l.append(np.array(new_t))
        new_c.append(np.array(new_l))
    return np.array(new_c)


def padding_commit_code(data, max_line, max_length):
    dict_code = dictionary_code(data)
    data_removed, data_added = [d['removed'] for d in data], [d['added'] for d in data]

    # padding commit code
    padding_removed_code = mapping_commit_code(data=data_removed, max_line=max_line, max_length=max_length)
    padding_removed_code = mapping_commit_dict_code(data=padding_removed_code, dict_code=dict_code)

    padding_added_code = mapping_commit_code(data=data_added, max_line=max_line, max_length=max_length)
    padding_added_code = mapping_commit_dict_code(data=padding_added_code, dict_code=dict_code)
    return padding_removed_code, padding_added_code, dict_code


def tokenize_commit_msg(data):
    tokens = [d.split() for d in data]
    tokens = list(itertools.chain.from_iterable(tokens))
    tokens = [t.lower() for t in tokens]
    return sorted(list(set(tokens)))


def commit_msg_label(data):
    dict_msg = tokenize_commit_msg(data=data)
    labels_ = np.array([1 if w in d.split() else 0 for d in data for w in dict_msg])
    labels_ = np.reshape(labels_, (int(labels_.shape[0] / len(dict_msg)), len(dict_msg)))
    return labels_, dict_msg


if __name__ == '__main__':
    # create padding for commit code
    ##################################################################################
    ##################################################################################
    # path_train_diff = './data/2017_ASE_Jiang/train.26208.diff'
    # data_train_diff = load_Jiang_code_data(pfile=path_train_diff)
    # path_test_diff = './data/2017_ASE_Jiang/test.3000.diff'
    # data_test_diff = load_Jiang_code_data(pfile=path_test_diff)
    # data_diff = data_train_diff + data_test_diff
    # print(len(data_diff))
    # max_line, max_length = 15, 40
    # padding_commit_code(data=data_diff, max_line=max_line, max_length=max_length)

    # create label using the commit message
    ##################################################################################
    ##################################################################################
    path_train_msg = './data/2017_ASE_Jiang/train.26208.msg'
    data_train_msg = load_file(path_file=path_train_msg)
    path_test_msg = './data/2017_ASE_Jiang/test.3000.msg'
    data_test_msg = load_file(path_file=path_test_msg)
    print(len(data_train_msg + data_test_msg))
    data = data_train_msg + data_test_msg
    label, dict_msg = commit_msg_label(data=data)
    print(label.shape, len(dict_msg))
