import numpy as np 

def clean_each_line(line):
    line = line.strip()
    line = line.lower().replace('/', ' ')
    line = line.split()
    return ' '.join(line).strip()


def get_diff_data(commit, type_code): 
    lines = list()
    for c in commit:
        c = clean_each_line(c.strip())
        if c.strip() != '':
            if type_code == 'added':
                if c.startswith('+'):
                    lines.append(c[1:].strip())
                elif c.startswith('-'):
                    # do nothing
                    pass
                else:
                    lines.append(c)
            elif type_code == 'removed':
                if c.startswith('-'):
                    lines.append(c[1:].strip())
                elif c.startswith('+'):
                    # do nothing
                    pass
                else:
                    lines.append(c)
    return lines

def check_diff_data(data):
    new_data = list()
    for d in data:
        lines = d.split('<nl>')
        for l in lines:
            l = clean_each_line(l.strip())            
            if l.startswith('+') or l.startswith('-'):
                new_data.append(d)
                break
    return new_data


def load_code_data(data):    
    data_removed = [get_diff_data(commit=d.split('<nl>'), type_code='removed') for d in data]
    data_added = [get_diff_data(commit=d.split('<nl>'), type_code='added') for d in data]
    data = [{'removed': r, 'added': a} for a, r in zip(data_added, data_removed)]
    return data

def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        return str(line + ' <NULL>' * (max_length - line_length)).strip()
    elif line_length > max_length:
        line_split = line.split()
        return " ".join([line_split[i] for i in range(max_length)])
    else:
        return line

def mapping_commit_code(data, max_line, max_length):
    padding_length = padding_commit_code_length(data=data, max_length=max_length)
    padding_line = padding_commit_code_line(padding_length, max_line=max_line, max_length=max_length)
    return padding_line

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

def mapping_commit_dict_code(data, dict_code):
    new_c = list()
    for c in data:
        new_l = list()
        for l in c:
            l_split, new_t = l.split(), list()
            for t in l_split:
                if t in dict_code.keys():
                    new_t.append(dict_code[t])
                else:
                    new_t.append(dict_code['<NULL>'])
            new_l.append(np.array(new_t))
        new_c.append(np.array(new_l))
    return np.array(new_c)

def padding_commit_code(data, max_line, max_length, dict_code):
    data_removed, data_added = [d['removed'] for d in data], [d['added'] for d in data]

    # padding commit code
    padding_removed_code = mapping_commit_code(data=data_removed, max_line=max_line, max_length=max_length)
    padding_removed_code = mapping_commit_dict_code(data=padding_removed_code, dict_code=dict_code)

    padding_added_code = mapping_commit_code(data=data_added, max_line=max_line, max_length=max_length)
    padding_added_code = mapping_commit_dict_code(data=padding_added_code, dict_code=dict_code)
    return padding_removed_code, padding_added_code

def processing_data(code, dictionary, params):
    code = load_code_data(code)
    dict_msg, dict_code = dictionary    
    max_line, max_length = params.code_line, params.code_length
    pad_removed_code, pad_added_code = padding_commit_code(data=code, max_line=max_line, max_length=max_length, dict_code=dict_code)
    return pad_added_code, pad_removed_code
