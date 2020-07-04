import numpy as np

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

def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        return str(line + ' <NULL>' * (max_length - line_length)).strip()
    elif line_length > max_length:
        line_split = line.split()
        return ' '.join([line_split[i] for i in range(max_length)])
    else:
        return line

def convert_msg_to_label(pad_msg, dict_msg):
    nrows, ncols = pad_msg.shape
    labels = list()
    for i in range(nrows):
        column = list(set(list(pad_msg[i, :])))
        label = np.zeros(len(dict_msg))
        for c in column:
            label[c] = 1
        labels.append(label)
    return np.array(labels)

def mapping_dict_msg(pad_msg, dict_msg):
    return np.array(
        [np.array([dict_msg[w.lower()] if w.lower() in dict_msg.keys() else dict_msg['<NULL>'] for w in line.split(' ')]) for line in pad_msg])

def mapping_dict_code(pad_code, dict_code):
    new_pad_code = list()
    for commit in pad_code:
        new_files = list()
        for file in commit:
            new_file = list()
            for line in file:
                new_line = list()
                for token in line.split(' '):
                    if token.lower() in dict_code.keys():
                        new_line.append(dict_code[token.lower()])
                    else:
                        new_line.append(dict_code['<NULL>'])
                new_file.append(np.array(new_line))
            new_file = np.array(new_file)
            new_files.append(new_file)
        new_files = np.array(new_files)
        new_pad_code.append(new_files)
    return np.array(new_pad_code)

def padding_commit_code(data, max_file, max_line, max_length):
    padding_length = padding_commit_code_length(data=data, max_length=max_length)
    padding_line = padding_commit_code_line(padding_length, max_line=max_line, max_length=max_length)
    padding_file = padding_commit_file(data=padding_line, max_file=max_file, max_line=max_line, max_length=max_length)
    return padding_file

def clean_and_reformat_code(data):
    # remove empty lines in code; divide code to two part: added_code and removed_code
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

def padding_message(data, max_length):
    new_data = list()
    for d in data:
        new_data.append(padding_length(line=d, max_length=max_length))
    return new_data