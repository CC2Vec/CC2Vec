from bfp_extracting import commit_id, commit_stable, commit_msg, commit_date, commit_code
from bfp_reformating import reformat_file, reformat_hunk
import numpy as np
import math
import os


def load_file(path_file):
    lines = list(open(path_file, "r", encoding="utf-8").readlines())
    lines = [l.strip() for l in lines]
    return lines


def load_dict_file(path_file):
    lines = list(open(path_file, "r").readlines())
    dictionary = dict()
    for line in lines:
        key, value = line.split('\t')[0], line.split('\t')[1]
        dictionary[key] = value
    return dictionary


def write_dict_file(path_file, dictionary):
    split_path = path_file.split("/")
    path_ = split_path[:len(split_path) - 1]
    path_ = "/".join(path_)

    if not os.path.exists(path_):
        os.makedirs(path_)

    with open(path_file, 'w') as out_file:
        for key in dictionary.keys():
            # write line to output file
            out_file.write(str(key) + '\t' + str(dictionary[key]))
            out_file.write("\n")
        out_file.close()


def write_file(path_file, data):
    split_path = path_file.split("/")
    path_ = split_path[:len(split_path) - 1]
    path_ = "/".join(path_)

    if not os.path.exists(path_):
        os.makedirs(path_)

    with open(path_file, 'w') as out_file:
        for line in data:
            # write line to output file
            out_file.write(str(line))
            out_file.write("\n")
        out_file.close()


def commits_index(commits):
    commits_index = [i for i, c in enumerate(commits) if c.startswith("commit:")]
    return commits_index


def commit_info(commit):
    id_ = commit_id(commit)
    stable = commit_stable(commit)
    date = commit_date(commit)
    msg = commit_msg(commit)
    code = commit_code(commit)
    return id_, stable, date, msg, code


def extract_commit(path_file):
    commits = load_file(path_file=path_file)
    indexes = commits_index(commits=commits)
    dicts = list()
    for i in range(0, len(indexes)):
        dict_ = {}
        if i == len(indexes) - 1:
            id_, stable, date, msg, code = commit_info(commits[indexes[i]:])
        else:
            id_, stable, date, msg, code = commit_info(commits[indexes[i]:indexes[i + 1]])
        dict_["id"] = id_
        dict_["stable"] = stable
        dict_["date"] = date
        dict_["msg"] = msg
        dict_["code"] = code
        dicts.append(dict_)
    return dicts


def reformat_commit_code(commits, num_file, num_hunk, num_loc, num_leng):
    commits = reformat_file(commits=commits, num_file=num_file)
    commits = reformat_hunk(commits=commits, num_hunk=num_hunk, num_loc=num_loc, num_leng=num_leng)
    return commits


def random_mini_batch(X_msg, X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X_msg = X_msg[permutation, :]
    shuffled_X_added = X_added_code[permutation, :, :, :]
    shuffled_X_removed = X_removed_code[permutation, :, :, :]
    if len(Y.shape) == 1:
        shuffled_Y = Y[permutation]
    else:
        shuffled_Y = Y[permutation, :]    

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        # mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        # mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches(X_msg, X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg = X_msg
    shuffled_X_added = X_added_code
    shuffled_X_removed = X_removed_code
    shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


if __name__ == "__main__":    
    path_data = "./data/linux/newres_funcalls_words_jul28.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nloc, nleng = 2, 5, 10, 120
    new_commits = reformat_commit_code(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nloc, num_leng=nleng)
