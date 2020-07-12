import pickle
import argparse
import numpy as np
from bfp_cc2ftr_train import train_model
from bfp_cc2ftr_extracted import extracted_cc2ftr
from bfp_preprocessing import reformat_commit_code
from bfp_padding import padding_commit

def read_args():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-train', action='store_true', help='training attention model')

    parser.add_argument('-train_data', type=str, default='./data/bfp/train.pkl', help='the directory of our training data')
    parser.add_argument('-test_data', type=str, default='./data/bfp/test.pkl', help='the directory of our testing data')
    parser.add_argument('-dictionary_data', type=str, default='./data/bfp/dict.pkl', help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='extracting features')
    parser.add_argument('-predict_data', type=str, help='the directory of our extracting data')
    parser.add_argument('-name', type=str, help='name of our output file')

    # Predicting our data
    parser.add_argument('-load_model', type=str, default=None, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('-msg_length', type=int, default=512, help='the length of the commit message')
    parser.add_argument('-code_file', type=int, default=2, help='the number of files in commit code')
    parser.add_argument('-code_hunk', type=int, default=5, help='the number of hunks in each file in commit code')
    parser.add_argument('-code_line', type=int, default=8, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=32, help='the length of each LOC of commit code')

    # Number of parameters for Attention model
    parser.add_argument('-embed_size', type=int, default=16, help='the dimension of embedding vector')
    parser.add_argument('-hidden_size', type=int, default=8, help='the number of nodes in hidden states of wordRNN, sentRNN, and hunkRNN')
    parser.add_argument('-hidden_units', type=int, default=256, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=50, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')

    # Model
    parser.add_argument('-data_type', type=str, default='all', help='type of model for learning')
    parser.add_argument('-model', type=str, default='model', help='names of our model')

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')

    # option to load model
    parser.add_argument('-datetime', type=str, default=None, help='date of model [default: None]')
    parser.add_argument('-start_epoch', type=int, default=None, help='starting epoch of loading model')
    parser.add_argument('-end_epoch', type=int, default=None, help='ending epoch of loading model')
    parser.add_argument('-step', type=int, default=None, help='jumping step of the epoch')
    return parser

if __name__ == '__main__':        
    params = read_args().parse_args()    
    if params.train is True:
        train_data = pickle.load(open(params.train_data, 'rb'))
        test_data = pickle.load(open(params.test_data, 'rb'))
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary

        data = train_data + test_data
        data = reformat_commit_code(commits=data, num_file=params.code_file, num_hunk=params.code_hunk, 
                                num_loc=params.code_line, num_leng=params.code_length)
        pad_msg, pad_added_code, pad_removed_code, labels = padding_commit(commits=data, dictionary=dictionary, params=params)            
        dict_msg, dict_code = dictionary  

        data = (pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code)  
        train_model(data=data, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the training process---------------------------')
        print('--------------------------------------------------------------------------------')
        exit()

    elif params.predict is True:
        data = pickle.load(open(params.predict_data, 'rb'))        
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary

        data = reformat_commit_code(commits=data, num_file=params.code_file, num_hunk=params.code_hunk, 
                                num_loc=params.code_line, num_leng=params.code_length)
        pad_msg, pad_added_code, pad_removed_code, labels = padding_commit(commits=data, dictionary=dictionary, params=params)
        
        params.batch_size = 8        
        data = (pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code)
        extracted_cc2ftr(data=data, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the extracting process-------------------------')
        print('--------------------------------------------------------------------------------')
        exit()
    else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        exit()
