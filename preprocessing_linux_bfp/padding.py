#  * This file is part of PatchNet, licensed under the terms of the GPL v2.
#  * See copyright.txt in the PatchNet source code for more information.
#  * The PatchNet source code can be obtained at
#  * https://github.com/hvdthong/PatchNetTool


import numpy as np
from extracting import extract_msg, extract_code, dictionary


def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        new_line = line + " NULL" * (max_length - line_length)
        return new_line.strip()
    elif line_length > max_length:
        line_split = line.split()
        return " ".join([line_split[i] for i in range(max_length)])
    else:
        return line


def padding_multiple_length(lines, max_length):
    new_lines = list()
    for l in lines:
        new_lines.append(padding_length(line=l, max_length=max_length))
    return new_lines


def mapping_commit_msg(msgs, max_length, dict_msg):
    pad_msg = padding_multiple_length(lines=msgs, max_length=max_length)
    new_pad_msg = list()
    for line in pad_msg:
        line_split = line.split(" ")
        new_line = list()
        for w in line_split:
            if w in dict_msg:
                new_line.append(dict_msg[w])
            else:
                new_line.append(dict_msg['NULL'])
        new_pad_msg.append(np.array(new_line))
    return np.array(new_pad_msg)


def filtering_code(lines):
    new_lines = list()
    for l in lines:
        code = " ".join(l.split(":")[1].split(","))
        code = l.split(":")[0] + " " + code
        new_lines.append(code)
    return new_lines


def padding_line(lines, max_line, max_length):
    new_lines = padding_multiple_length(lines=lines, max_length=max_length)
    if len(lines) < max_line:
        for l in range(0, max_line - len(lines)):
            new_lines.append(padding_length(line="", max_length=max_length))
        return new_lines
    elif len(lines) > max_line:
        return [new_lines[i] for i in range(max_line)]
    else:
        return new_lines


def padding_hunk_code(code, max_hunk, max_line, max_length):
    new_hunks = dict()
    for i in range(1, max_hunk + 1):
        if i not in code.keys():
            new_hunks[i] = padding_line(lines=[""], max_line=max_line, max_length=max_length)
        else:
            new_hunks[i] = padding_line(lines=filtering_code(code[i]), max_line=max_line, max_length=max_length)
    return new_hunks


def padding_hunk(file, max_hunk, max_line, max_length):
    new_file = dict()
    new_file["removed"] = padding_hunk_code(file["removed"], max_hunk=max_hunk, max_line=max_line,
                                            max_length=max_length)
    new_file["added"] = padding_hunk_code(file["added"], max_hunk=max_hunk, max_line=max_line, max_length=max_length)
    return new_file


def padding_file(commits, max_hunk, max_line, max_length):
    # remember that we assume that we only have one file in commit code
    padding_code = list()
    for c in commits:
        files = c["code"]
        pad_file = list()
        for f in files:
            pad_file.append(padding_hunk(file=f, max_hunk=max_hunk, max_line=max_line, max_length=max_length))
        padding_code.append(pad_file)
    return padding_code


def mapping_commit_code_file(code, dict_code):
    new_hunks = list()
    for k in code.keys():
        hunk, new_hunk = code[k], list()
        for l in hunk:
            split_ = l.split(" ")
            new_line = list()
            for w in split_:
                if w in dict_code:
                    new_line.append(dict_code[w])
                else:
                    new_line.append(dict_code['NULL'])
            new_hunk.append(np.array(new_line))
        new_hunks.append(np.array(new_hunk))
    return np.array(new_hunks)


def mapping_commit_code(type, commits, max_hunk, max_code_line, max_code_length, dict_code):
    pad_code = padding_file(commits=commits, max_hunk=max_hunk, max_line=max_code_line, max_length=max_code_length)
    new_pad_code = list()
    for p in pad_code:
        file_ = p[0]  # we only use one file
        new_file = mapping_commit_code_file(code=file_[type], dict_code=dict_code)
        new_pad_code.append(new_file)
    return np.array(new_pad_code)


def load_label_commits(commits):
    labels = [1 if c["stable"] == "true" else 0 for c in commits]
    return np.array(labels, dtype=np.float)


###########################################################################
###########################################################################
def padding_commit_topwords(commits, params):
    codes = extract_code(commits=commits)
    dict_code = dictionary(data=codes)

    # padding commit code
    pad_added_code = mapping_commit_code(type="added", commits=commits, max_hunk=params.code_hunk,
                                         max_code_line=params.code_line,
                                         max_code_length=params.code_length, dict_code=dict_code)
    pad_removed_code = mapping_commit_code(type="removed", commits=commits, max_hunk=params.code_hunk,
                                           max_code_line=params.code_line,
                                           max_code_length=params.code_length, dict_code=dict_code)
    return pad_added_code, pad_removed_code, dict_code


def padding_label_topwords(commits, topwords):
    labels_ = np.array([1 if w in c['msg'].split(',') else 0 for c in commits for w in topwords])
    labels_ = np.reshape(labels_, (int(labels_.shape[0] / len(topwords)), len(topwords)))
    return labels_


def padding_commit(commits, params):
    msgs, codes = extract_msg(commits=commits), extract_code(commits=commits)
    dict_msg, dict_code = dictionary(data=msgs), dictionary(data=codes)

    # padding commit message
    pad_msg = mapping_commit_msg(msgs=msgs, max_length=params.msg_length, dict_msg=dict_msg)
    # padding commit code
    pad_added_code = mapping_commit_code(type="added", commits=commits, max_hunk=params.code_hunk,
                                         max_code_line=params.code_line,
                                         max_code_length=params.code_length, dict_code=dict_code)
    pad_removed_code = mapping_commit_code(type="removed", commits=commits, max_hunk=params.code_hunk,
                                           max_code_line=params.code_line,
                                           max_code_length=params.code_length, dict_code=dict_code)
    labels = load_label_commits(commits=commits)
    return pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code


def padding_pred_commit(commits, params, dict_msg, dict_code):
    msgs, codes = extract_msg(commits=commits), extract_code(commits=commits)

    # padding commit message
    pad_msg = mapping_commit_msg(msgs=msgs, max_length=params.msg_length, dict_msg=dict_msg)
    # padding commit code
    pad_added_code = mapping_commit_code(type="added", commits=commits, max_hunk=params.code_hunk,
                                         max_code_line=params.code_line,
                                         max_code_length=params.code_length, dict_code=dict_code)
    pad_removed_code = mapping_commit_code(type="removed", commits=commits, max_hunk=params.code_hunk,
                                           max_code_line=params.code_line,
                                           max_code_length=params.code_length, dict_code=dict_code)
    labels = load_label_commits(commits=commits)
    return pad_msg, pad_added_code, pad_removed_code, labels
