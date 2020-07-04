from bfp_cc2ftr_model import HierachicalRNN
from bfp_utils import mini_batches, convert_msg_to_label
import torch
from tqdm import tqdm
import pickle

def extracted_cc2ftr(data, params):
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)
    batches = mini_batches(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=pad_msg_labels, 
                            mini_batch_size=params.batch_size, shuffled=False)    
            
    params.vocab_code = len(dict_code)
    if len(pad_msg_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = pad_msg_labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierachicalRNN(args=params)
    model.load_state_dict(torch.load(params.load_model))

    if torch.cuda.is_available():
        model = model.cuda()
    commit_ftrs = list()
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
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
    print(commit_ftrs.shape)
    pickle.dump(commit_ftrs, open(params.name, 'wb'))
    exit()