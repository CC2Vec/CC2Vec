from parameters import read_args_jiang
from jiang_prepared import load_Jiang_code_data
from ultis import load_file
from jiang_train import train_model

if __name__ == '__main__':
    input_option = read_args_jiang().parse_args()
    input_help = read_args_jiang().print_help()

    # loading the commit code
    ##################################################################################
    ##################################################################################
    path_train_diff = './data/jiang_ase_2017/train.26208.diff'
    data_train_diff = load_Jiang_code_data(pfile=path_train_diff)
    path_test_diff = './data/jiang_ase_2017/test.3000.diff'
    data_test_diff = load_Jiang_code_data(pfile=path_test_diff)
    data_diff = data_train_diff + data_test_diff
    padding_code_info = (15, 40)  # max_line = 15; max_length = 40
    ##################################################################################
    ##################################################################################

    # loading the commit msg
    ##################################################################################
    ##################################################################################
    path_train_msg = './data/jiang_ase_2017/train.26208.msg'
    data_train_msg = load_file(path_file=path_train_msg)
    path_test_msg = './data/jiang_ase_2017/test.3000.msg'
    data_test_msg = load_file(path_file=path_test_msg)
    data_msg = data_train_msg + data_test_msg
    ##################################################################################
    ##################################################################################
    train_model(commit_diff=data_diff, commit_msg=data_msg, params=input_option,
                padding_code_info=padding_code_info)
