from jit_DExtended_model import DeepJITExtended
import torch 
from tqdm import tqdm
from jit_utils import mini_batches_update_DExtended
import torch.nn as nn
import os, datetime
from jit_utils import save

def train_model(data, params):
    cc2ftr, data_pad_msg, data_pad_code, data_labels, dict_msg, dict_code = data
    
    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    params.embedding_ftr = cc2ftr.shape[1]

    if len(data_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = data_labels.shape[1]
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create and train the defect model
    model = DeepJITExtended(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)

    criterion = nn.BCELoss()
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        # building batches for training model
        batches = mini_batches_update_DExtended(X_ftr=cc2ftr, X_msg=data_pad_msg, X_code=data_pad_code, Y=data_labels)
        for i, (batch) in enumerate(tqdm(batches)):
            ftr, pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                ftr = torch.tensor(ftr).cuda()
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                ftr = torch.tensor(ftr).long()
                pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()
            predict = model.forward(ftr, pad_msg, pad_code)
            loss = criterion(predict, labels)
            total_loss += loss
            loss.backward()
            optimizer.step()

        print('Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))    
        save(model, params.save_dir, 'epoch', epoch)
