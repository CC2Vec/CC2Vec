from ultis import extract_commit
from ultis import reformat_commit_code
import pandas as pd
import pickle


def get_commit_id(commit):
    return [c['id'].strip() for c in commit]


def get_commit_label(commit):
    return [c['stable'].strip() for c in commit]


if __name__ == '__main__':
    path_data = "../data/linux/newres_funcalls_jul28.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nloc, nleng = 1, 8, 10, 120

    commits = reformat_commit_code(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nloc, num_leng=nleng)
    commits = commits[:int(len(commits) / 64) * 64]
    commits_id = get_commit_id(commit=commits)
    commits_label = get_commit_label(commit=commits)

    path_raw_features = '../data/linux/raw_features.txt'
    raw_ftr = pd.read_csv(path_raw_features, header=None)
    raw_ftr_id = list(raw_ftr.iloc[:, 0])
    raw_ftr_id = [id.strip() for id in raw_ftr_id]

    interset_id = list(set(commits_id) & set(raw_ftr_id))
    rows, indexes = list(), list()

    for id in interset_id:
        row = raw_ftr.loc[raw_ftr[0] == id]
        rows.append(row)
        print(len(rows))
        indexes.append(commits_id.index(id))
    raw_ftr = pd.concat(rows)
    data = (indexes, raw_ftr)
    print(len(indexes), len(raw_ftr))

    # raw_ftr.to_csv('../data/bfp_linux_raw_features.csv', header=False, index=False)

    with open('../data/bfp_raw_features.pickle', 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
