#  * This file is part of PatchNet, licensed under the terms of the GPL v2.
#  * See copyright.txt in the PatchNet source code for more information.
#  * The PatchNet source code can be obtained at
#  * https://github.com/hvdthong/PatchNetTool


def reformat_file(commits, num_file):
    for c in commits:
        if len(c['code']) > num_file:
            code_files = c['code']
            c.update({'code': [code_files[0]]})
    return commits


def update_hunk(hunk, num_hunk, num_loc, num_leng):
    new_hunk = dict()
    for key in hunk:
        if key <= num_hunk:
            loc_values = hunk[key][:num_loc]
            length_values = list()
            for v in loc_values:
                split_v = v.split(',')[:num_leng]
                length_values.append(','.join(split_v))
            new_hunk[key] = length_values
    return new_hunk


def reformat_hunk(commits, num_hunk, num_loc, num_leng):
    for c in commits:
        hunk = c['code'][0]
        new_added_hunk = update_hunk(hunk=hunk['added'], num_hunk=num_hunk, num_loc=num_loc, num_leng=num_leng)
        new_removed_hunk = update_hunk(hunk=hunk['removed'], num_hunk=num_hunk, num_loc=num_loc, num_leng=num_leng)
        hunk.update({'added': new_added_hunk})
        hunk.update({'removed': new_removed_hunk})
    return commits
