import argparse
from lmg_padding import processing_data
import pickle
from lmg_cc2ftr_train import train_model
from lmg_cc2ftr_extracted import extracted_cc2ftr

def read_args_lmg():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-train', action='store_true', help='training PatchNet model')
    parser.add_argument('-train_data', type=str, default='./data/lmg/train.pkl', help='the directory of our training data')
    parser.add_argument('-dictionary_data', type=str, default='./data/lmg/dict.pkl', help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str, help='the directory of our training data')        
    parser.add_argument('-name', type=str, help='name of our output file')

    # Predicting our data
    parser.add_argument('-load_model', type=str, default=None, help='loading our model')

    # Number of parameters for reformatting commits        
    parser.add_argument('-code_line', type=int, default=15, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=40, help='the length of each LOC of commit code')

    # Number of parameters for Attention model
    parser.add_argument('-embed_size', type=int, default=16, help='the dimension of embedding vector')
    parser.add_argument('-hidden_size', type=int, default=8, help='the number of nodes in hidden states of wordRNN, sentRNN, and hunkRNN')
    parser.add_argument('-hidden_units', type=int, default=256, help='the number of nodes in hidden layers')
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
    params = read_args_lmg().parse_args()    
    if params.train is True:
        train_data = pickle.load(open(params.train_data, 'rb'))
        train_msg, train_diff = train_data[0], train_data[1]
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))        
        train_pad_added_code, train_pad_removed_code = processing_data(code=train_diff, dictionary=dictionary, params=params)
        dict_msg, dict_code = dictionary        

        data = (train_msg, train_pad_added_code, train_pad_removed_code, dict_msg, dict_code)  
        train_model(data=data, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the training process---------------------------')
        print('--------------------------------------------------------------------------------')
        exit()

    elif params.predict is True:
        data = pickle.load(open(params.pred_data, 'rb'))
        msg, diff = data[0], data[1]       
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))        
        pad_added_code, pad_removed_code = processing_data(code=diff, dictionary=dictionary, params=params)
        dict_msg, dict_code = dictionary

        data = (msg, pad_added_code, pad_removed_code, dict_msg, dict_code)  
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

