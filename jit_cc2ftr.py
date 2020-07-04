import argparse
import pickle
import numpy as np 
from jit_padding import padding_message, clean_and_reformat_code, padding_commit_code, mapping_dict_msg, mapping_dict_code, convert_msg_to_label
from jit_cc2ftr_train import train_model
from jit_cc2ftr_extracted import extracted_cc2ftr

def read_args():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-project', type=str, default='openstack', help='name of the dataset')

    # Training our model
    parser.add_argument('-train', action='store_true', help='training attention model')

    parser.add_argument('-train_data', type=str, default='./data/jit/openstack_train.pkl', help='the directory of our training data')
    parser.add_argument('-test_data', type=str, default='./data/jit/openstack_test.pkl', help='the directory of our testing data')
    parser.add_argument('-dictionary_data', type=str, default='./data/jit/openstack_dict.pkl', help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='extracting features')
    parser.add_argument('-predict_data', type=str, help='the directory of our extracting data')
    parser.add_argument('-name', type=str, help='name of our output file')

    # Predicting our data
    parser.add_argument('-load_model', type=str, default=None, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('--msg_length', type=int, default=256, help='the length of the commit message')
    parser.add_argument('--code_file', type=int, default=2, help='the number of files in commit code')
    parser.add_argument('--code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('--code_length', type=int, default=64, help='the length of each LOC of commit code')

    # Predicting our data
    parser.add_argument('--predict', action='store_true', help='predicting testing data')

    # Number of parameters for Attention model
    parser.add_argument('-embed_size', type=int, default=64, help='the dimension of embedding vector')
    parser.add_argument('-hidden_size', type=int, default=32, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training PatchNet')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=50, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')    

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()    
    
    if params.train is True:
        train_data = pickle.load(open(params.train_data, 'rb'))
        train_ids, train_labels, train_messages, train_codes = train_data    

        test_data = pickle.load(open(params.test_data, 'rb'))
        test_ids, test_labels, test_messages, test_codes = test_data        

        ids = train_ids + test_ids
        labels = list(train_labels) + list(test_labels)
        msgs = train_messages + test_messages
        codes = train_codes + test_codes
        
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary  

        pad_msg = padding_message(data=msgs, max_length=params.msg_length)
        added_code, removed_code = clean_and_reformat_code(codes)
        pad_added_code = padding_commit_code(data=added_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)
        pad_removed_code = padding_commit_code(data=removed_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)

        pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
        pad_added_code = mapping_dict_code(pad_code=pad_added_code, dict_code=dict_code)
        pad_removed_code = mapping_dict_code(pad_code=pad_removed_code, dict_code=dict_code)
        pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)

        print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))
        print('Shape of commit message:', pad_msg.shape)
        print('Shape of added code:', pad_added_code.shape)
        print('Shape of removed code:', pad_removed_code.shape)
        print('Shape of message labels:', pad_msg_labels.shape)        

        data = (pad_added_code, pad_removed_code, pad_msg_labels, dict_msg, dict_code)   
        train_model(data=data, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the training process---------------------------')
        print('--------------------------------------------------------------------------------')
        exit()
    
    elif params.predict is True:
        data = pickle.load(open(params.predict_data, 'rb'))
        ids, labels, msgs, codes = data 

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))   
        dict_msg, dict_code = dictionary  

        pad_msg = padding_message(data=msgs, max_length=params.msg_length)
        added_code, removed_code = clean_and_reformat_code(codes)
        pad_added_code = padding_commit_code(data=added_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)
        pad_removed_code = padding_commit_code(data=removed_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)

        pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
        pad_added_code = mapping_dict_code(pad_code=pad_added_code, dict_code=dict_code)
        pad_removed_code = mapping_dict_code(pad_code=pad_removed_code, dict_code=dict_code)
        pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)

        print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))
        print('Shape of commit message:', pad_msg.shape)
        print('Shape of added code:', pad_added_code.shape)
        print('Shape of removed code:', pad_removed_code.shape)
        print('Shape of message labels:', pad_msg_labels.shape)        

        data = (pad_added_code, pad_removed_code, pad_msg_labels, dict_msg, dict_code)   
        extracted_cc2ftr(data=data, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the extracting process-------------------------')
        print('--------------------------------------------------------------------------------')
        exit()
