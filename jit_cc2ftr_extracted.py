from jit_cc2ftr_model import HierachicalRNN
from jit_utils import mini_batches
import torch
from tqdm import tqdm
import pickle

def extracted_cc2ftr(data, params):
    pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data    
    batches = mini_batches(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels, 
                            mini_batch_size=params.batch_size, shuffled=False)  
    params.vocab_code = len(dict_code)    
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierachicalRNN(args=params)
    model.load_state_dict(torch.load(params.load_model))
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    commit_ftrs = list()
    with torch.no_grad():
        for i, (batch) in enumerate(tqdm(batches)):            
            state_word = model.init_hidden_word()
            state_sent = model.init_hidden_sent()
            state_hunk = model.init_hidden_hunk()

            pad_added_code, pad_removed_code, labels = batch
            labels = torch.cuda.FloatTensor(labels)
            commit_ftr = model.forward_commit_embeds_diff(pad_added_code, pad_removed_code, state_hunk, state_sent, state_word)            
            commit_ftrs.append(commit_ftr)
        commit_ftrs = torch.cat(commit_ftrs).cpu().detach().numpy()
    pickle.dump(commit_ftrs, open(params.name, 'wb'))