from clean_commit import loading_variable
import numpy as np


def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        return str(line + ' <NULL>' * (max_length - line_length)).strip()
    elif line_length > max_length:
        line_split = line.split()
        return ' '.join([line_split[i] for i in range(max_length)])
    else:
        return line


def padding_message(data, max_length):
    new_data = list()
    for d in data:
        new_data.append(padding_length(line=d, max_length=max_length))
    return new_data


def padding_multiple_length(lines, max_length):
    return [padding_length(line=l, max_length=max_length) for l in lines]


def padding_commit_code_length(data, max_length):
    commits = list()
    for commit in data:
        new_commit = list()
        for file in commit:
            new_file = list()
            for line in file:
                new_line = padding_length(line, max_length=max_length)
                new_file.append(new_line)
            new_commit.append(new_file)
        commits.append(new_commit)
    return commits


def padding_commit_code_line(data, max_line, max_length):
    new_commits = list()
    for commit in data:
        new_files = list()
        for file in commit:
            new_file = file
            if len(file) == max_line:
                new_file = file
            elif len(file) > max_line:
                new_file = file[:max_line]
            else:
                num_added_line = max_line - len(file)
                new_file = file
                for i in range(num_added_line):
                    new_file.append(('<NULL> ' * max_length).strip())
            new_files.append(new_file)
        new_commits.append(new_files)
    return new_commits


def padding_commit_file(data, max_file, max_line, max_length):
    new_commits = list()
    for commit in data:
        new_commit = list()
        if len(commit) == max_file:
            new_commit = commit
        elif len(commit) > max_file:
            new_commit = commit[:max_file]
        else:
            num_added_file = max_file - len(commit)
            new_files = list()
            for i in range(num_added_file):
                file = list()
                for j in range(max_line):
                    file.append(('<NULL> ' * max_length).strip())
                new_files.append(file)
            new_commit = commit + new_files
        new_commits.append(new_commit)
    return new_commits


def padding_commit_code(data, max_file, max_line, max_length):
    padding_length = padding_commit_code_length(data=data, max_length=max_length)
    padding_line = padding_commit_code_line(padding_length, max_line=max_line, max_length=max_length)
    padding_file = padding_commit_file(data=padding_line, max_file=max_file, max_line=max_line, max_length=max_length)
    return padding_file


def dictionary_commit(data, type_data):
    # create dictionary for commit message
    lists = list()
    if type_data == 'msg':
        for m in data:
            lists += m.split()
    elif type_data == 'code':
        added_code, removed_code = data
        for diff in added_code:
            for file in diff:
                for line in file:
                    lists += line.split()
        for diff in removed_code:
            for file in diff:
                for line in file:
                    lists += line.split()
    else:
        print('You need to give an correct data type')
        exit()
    lists = list(sorted(list(set(lists))))
    lists.append('<NULL>')
    new_dict = dict()
    for i in range(len(lists)):
        new_dict[lists[i]] = i
    return new_dict


def mapping_dict_msg(pad_msg, dict_msg):
    return np.array(
        [np.array([dict_msg[w] if w in dict_msg else dict_msg['<NULL>'] for w in line.split(' ')]) for line in pad_msg])


def mapping_dict_code(pad_code, dict_code):
    new_pad_code = list()
    for commit in pad_code:
        new_files = list()
        for file in commit:
            new_file = list()
            for line in file:
                new_line = list()
                for token in line.split(' '):
                    if token in dict_code:
                        new_line.append(dict_code[token])
                    else:
                        new_line.append(dict_code['<NULL>'])
                new_file.append(np.array(new_line))
            new_file = np.array(new_file)
            new_files.append(new_file)
        new_files = np.array(new_files)
        new_pad_code.append(new_files)
    return np.array(new_pad_code)


if __name__ == '__main__':
    project = 'openstack'
    messages, codes = loading_variable(project + '_messages'), loading_variable(project + '_codes')
    print('Number of instances in commit message %i and commit code %i ' % (len(messages), len(codes)))
    dict_msg, dict_code = dictionary_commit(data=messages, type_data='msg'), dictionary_commit(data=codes,
                                                                                               type_data='code')
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))
    pad_msg, pad_code = padding_message(data=messages, max_length=256), padding_commit_code(data=codes, max_line=10,
                                                                                            max_length=512)
    pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
    pad_code = mapping_dict_code(pad_code=pad_code, dict_code=dict_code)
    print('Shape of commit messages: ', pad_msg.shape)
    print('Shape of commit code: ', pad_code.shape)
