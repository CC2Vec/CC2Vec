from bfp_utils import mini_batches, convert_msg_to_label
from bfp_cc2ftr_model import HierachicalRNN
import torch 
import torch.nn as nn
from tqdm import tqdm
import os, datetime
from bfp_utils import save

def train_model(data, params):
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)
    batches = mini_batches(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels,
                           mini_batch_size=params.batch_size)

    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)

    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
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