import argparse
import pickle
from bfp_PNExtended_train import train_model
from bfp_PNExtended_eval import evaluation_model
from bfp_preprocessing import reformat_commit_code
from bfp_padding import padding_commit

def read_args_PNExtend():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-train', action='store_true', help='training PatchNet model')    

    parser.add_argument('-train_data', type=str, default='./data/bfp/train.pkl', help='the directory of our training data')
    parser.add_argument('-train_data_cc2ftr', type=str, default='./data/bfp/train_cc2ftr.pkl', help='the directory of our training data with cc2ftr')
    parser.add_argument('-dictionary_data', type=str, default='./data/bfp/dict.pkl', help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str, default='./data/bfp/test.pkl', help='the directory of our testing data')
    parser.add_argument('-pred_data_cc2ftr', type=str, default='./data/bfp/test_cc2ftr.pkl', help='the directory of our testing data with cc2ftr')

    # Predicting our data
    parser.add_argument('-load_model', type=str, default='./model/bfp/pn_extend.pt', help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('--msg_length', type=int, default=512, help='the length of the commit message')
    parser.add_argument('--code_file', type=int, default=2, help='the number of files in commit code')
    parser.add_argument('--code_hunk', type=int, default=5, help='the number of hunks in each file in commit code')
    parser.add_argument('--code_line', type=int, default=8, help='the number of LOC in each hunk of commit code')
    parser.add_argument('--code_length', type=int, default=32, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('--embedding_dim', type=int, default=8, help='the dimension of embedding vector')
    parser.add_argument('--filter_sizes', type=str, default='1, 2', help='the filter size of convolutional layers')
    parser.add_argument('--num_filters', type=int, default=32, help='the number of filters')
    parser.add_argument('--hidden_units', type=int, default=128, help='the number of nodes in hidden layers')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='dropout for training PatchNet')
    parser.add_argument('--l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')

    # Config tensorflow
    parser.add_argument('--allow_soft_placement', type=bool, default=True, help='allow device soft device placement')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log placement of ops on devices')
    return parser

if __name__ == '__main__':
    params = read_args_PNExtend().parse_args()    
    if params.train is True:
        train_data = pickle.load(open(params.train_data, 'rb'))
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary        

        train_data = reformat_commit_code(commits=train_data, num_file=params.code_file, num_hunk=params.code_hunk, 
                                num_loc=params.code_line, num_leng=params.code_length)
        train_pad_msg, train_pad_added_code, train_pad_removed_code, train_labels = padding_commit(commits=train_data, dictionary=dictionary, params=params)  
        train_data_cc2ftr = pickle.load(open(params.train_data_cc2ftr, 'rb'))        
        
        data = (train_data_cc2ftr, train_pad_msg, train_pad_added_code, train_pad_removed_code, train_labels, dict_msg, dict_code)  
        train_model(data=data, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the training process---------------------------')
        print('--------------------------------------------------------------------------------')
        exit()
    elif params.predict is True:
        pred_data = pickle.load(open(params.pred_data, 'rb'))
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary       

        pred_data = reformat_commit_code(commits=pred_data, num_file=params.code_file, num_hunk=params.code_hunk, 
                                num_loc=params.code_line, num_leng=params.code_length)
        pred_pad_msg, pred_pad_added_code, pred_pad_removed_code, pred_labels = padding_commit(commits=pred_data, dictionary=dictionary, params=params)           
        pred_data_cc2ftr = pickle.load(open(params.pred_data_cc2ftr, 'rb'))

        
        data = (pred_data_cc2ftr, pred_pad_msg, pred_pad_added_code, pred_pad_removed_code, pred_labels, dict_msg, dict_code)  
        evaluation_model(data=data, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the evaluation process---------------------------')
        print('--------------------------------------------------------------------------------')
        exit()
    else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        exit()