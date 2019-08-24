from ultis import load_file


def get_diff_data(commit, type_code):
    lines = list()
    for c in commit:
        c = c.strip()
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


def load_Jiang_code_data(pfile):
    data = load_file(path_file=pfile)
    data_removed = [get_diff_data(commit=d.split('<nl>'), type_code='removed') for d in data]
    data_added = [get_diff_data(commit=d.split('<nl>'), type_code='added') for d in data]
    data = [{'removed': r, 'added': a} for a, r in zip(data_added, data_removed)]
    return data


if __name__ == '__main__':
    path_train_diff = './data/2017_ASE_Jiang/train.26208.diff'
    data_train_diff = load_Jiang_code_data(pfile=path_train_diff)
    path_test_diff = './data/2017_ASE_Jiang/test.3000.diff'
    data_test_diff = load_Jiang_code_data(pfile=path_test_diff)
    data_diff = data_train_diff + data_test_diff
    print(len(data_diff))
    exit()
