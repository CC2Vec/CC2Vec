from lmg_utils import mini_batches, commit_msg_label
import os
import datetime
import torch.nn as nn
from tqdm import tqdm
import torch
from lmg_cc2ftr_model import HierachicalRNN
from lmg_utils import save


def train_model(data, params):
    msg, pad_added_code, pad_removed_code, dict_msg, dict_code = data
    labels = commit_msg_label(data=msg, dict_msg=dict_msg)
    batches = mini_batches(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels, mini_batch_size=params.batch_size)

    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)
    
    params.class_num = batches[0][2].shape[1]
    params.code_lines = batches[0][0].shape[1]

    # # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierachicalRNN(args=params)
    
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCEWithLogitsLoss()    
    for epoch in range(1, params.num_epochs + 1):        
        total_loss = 0
        for i, (batch) in enumerate(tqdm(batches)):            
            # reset the hidden state of hierarchical attention model
            state_word = model.init_hidden_word()
            state_sent = model.init_hidden_sent()
            state_hunk = model.init_hidden_hunk()

            pad_added_code, pad_removed_code, label = batch
            label = torch.cuda.FloatTensor(label)
            optimizer.zero_grad()
            predict = model.forward(pad_added_code, pad_removed_code, state_hunk, state_sent, state_word)
            loss = criterion(predict, label)
            loss.backward()
            total_loss += loss
            optimizer.step()

        print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))                
        save(model, params.save_dir, 'epoch', epoch)