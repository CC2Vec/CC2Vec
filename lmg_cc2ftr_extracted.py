from lmg_utils import mini_batches, commit_msg_label
from tqdm import tqdm
import torch
from lmg_cc2ftr_model import HierachicalRNN
import pickle

def extracted_cc2ftr(data, params):
    msg, pad_added_code, pad_removed_code, dict_msg, dict_code = data
    labels = commit_msg_label(data=msg, dict_msg=dict_msg)
    batches = mini_batches(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels, mini_batch_size=params.batch_size, Shuffled=False)

    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
        
    params.vocab_code = len(dict_code)
    
    params.class_num = batches[0][2].shape[1]
    params.code_lines = batches[0][0].shape[1]

    # # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierachicalRNN(args=params)
    model.load_state_dict(torch.load(params.load_model))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():        
        commit_ftrs = list()
        for i, (batch) in enumerate(tqdm(batches)):
            state_word = model.init_hidden_word()
            state_sent = model.init_hidden_sent()
            state_hunk = model.init_hidden_hunk()

            pad_added_code, pad_removed_code, label = batch            
            commit_ftr = model.forward_commit_embeds_diff(pad_added_code, pad_removed_code, state_hunk, state_sent, state_word)                
            commit_ftrs.append(commit_ftr)
        commit_ftrs = torch.cat(commit_ftrs).cpu().detach().numpy()    
    pickle.dump(commit_ftrs, open(params.name, 'wb'))