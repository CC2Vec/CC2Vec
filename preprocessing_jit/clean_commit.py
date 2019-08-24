from os import listdir
from os.path import isfile, join
from parser_commit import get_ids, info_commit
from ultis import load_file
from nltk.tokenize import word_tokenize
import string
import pickle


def clean_message(data):
    return [' '.join(word_tokenize(d['title'] + ' ' + d['desc'])).lower() for d in data]


def clean_code_line(line):
    for p in string.punctuation:
        if line.startswith('+#') or line.startswith('-#'):
            line = line[2:].replace(p, ' ' + p + ' ')
        elif line.startswith('+') or line.startswith('-') or line.startswith('#'):
            line = line[1:].replace(p, ' ' + p + ' ')
        else:
            line = line.replace(p, ' ' + p + ' ')
    return line


def clean_code(data):
    new_diffs = list()
    for diff in data:
        new_diff = list()
        for file_ in diff:
            lines = [' '.join(word_tokenize(clean_code_line(line=line))) for line in file_['diff']]
            new_diff.append(' '.join(word_tokenize(' '.join(lines).strip())))
        new_diffs.append(new_diff)
        print(len(new_diffs))
    return new_diffs


def clean_code_ver2(data):
    new_diffs = list()
    for diff in data:
        new_files = list()
        for file_ in diff:
            new_file, diff_code = dict(), file_['diff']
            added_code = [' '.join(word_tokenize(clean_code_line(line=line))) for line in diff_code['added_code']]
            removed_code = [' '.join(word_tokenize(clean_code_line(line=line))) for line in diff_code['removed_code']]
            new_file['added_code'], new_file['removed_code'] = added_code, removed_code
            new_files.append(new_file)
        new_diffs.append(new_files)
        print(len(new_diffs))
    return new_diffs


def collect_labels(path_data, path_label):
    valid_ids = get_ids([f for f in listdir(path_data) if isfile(join(path_data, f))])
    ids, labels = [l.split('\t')[0] for l in load_file(path_file=path_label)], [l.split('\t')[1] for l in
                                                                                load_file(path_file=path_label)]
    labels_valid_ids = [labels[ids.index(v_id)] for v_id in valid_ids if v_id in ids]
    return valid_ids, labels_valid_ids


def collect_labels_ver2(path_label):
    ids, labels = [l.split('\t')[0] for l in load_file(path_file=path_label)], [l.split('\t')[1] for l in
                                                                                load_file(path_file=path_label)]
    return ids, labels


def saving_variable(pname, variable):
    f = open('./variables/' + pname + '.pkl', 'wb')
    pickle.dump(variable, f)
    f.close()


def loading_variable(pname):
    f = open('./variables/' + pname + '.pkl', 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


if __name__ == '__main__':
    # project = 'openstack'
    project = 'qt'
    path_data = '../data/jit_defect/output/' + project
    path_label = '../data/jit_defect/labels/' + project + '_ids_label.txt'

    # ids, labels = collect_labels(path_data=path_data, path_label=path_label)
    ids, labels = collect_labels_ver2(path_label=path_label)

    messages, codes = info_commit(ids=ids, path_file=path_data)
    print(len(ids), len(labels), len(messages), len(codes))
    # messages, codes = clean_message(data=messages[:100]), clean_code(data=codes[:100])
    messages, codes = clean_message(data=messages), clean_code_ver2(data=codes)

    data = (messages, codes, labels, ids)
    write_data = open('../data/jit_defect/' + project + '.pkl', 'wb')
    pickle.dump(data, write_data)

    # saving_variable(project + '_messages', messages)
    # saving_variable(project + '_codes', codes)
    # saving_variable(project + '_labels', labels)
    # saving_variable(project + '_ids', ids)
