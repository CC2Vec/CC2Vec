from bfp_utils import mini_batches_PNExtended
import torch
from bfp_PNExtended_model import PatchNetExtended
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, roc_auc_score
import numpy as np 

def best_accuracy(true_label, pred_proba):
    
    fpr, tpr, thresholds = roc_curve(true_label, pred_proba)
    precision, recall, thresholds = precision_recall_curve(true_label, pred_proba)
    
    num_pos_class = len([1 for l in true_label if l == 1])
    num_neg_class = len([0 for l in true_label if l == 0])
    
    tp = recall * num_pos_class
    fp = (tp / precision) - tp
    tn = num_neg_class - fp
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]
    return np.amax(acc), best_threshold

def running_evaluation(model, data):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():        
        predicts, groundtruth = list(), list()
        for i, (batch) in enumerate(tqdm(data)):   
            embedding_ftr, pad_msg, pad_added_code, pad_removed_code, labels = batch
            embedding_ftr = torch.tensor(embedding_ftr).cuda()
            pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(pad_added_code).cuda(), torch.tensor(pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)            
            predicts.append(model.forward(embedding_ftr, pad_msg, pad_added_code, pad_removed_code))
            groundtruth.append(labels)
            
        predicts = torch.cat(predicts).cpu().detach().numpy()        
        groundtruth = torch.cat(groundtruth).cpu().detach().numpy()
        
        accuracy, _ = best_accuracy(groundtruth, predicts)
        binary_pred = [1 if p >= 0.5 else 0 for p in predicts]  # threshold can be changed 
        
        prc = precision_score(y_true=groundtruth, y_pred=binary_pred)        
        rc = recall_score(y_true=groundtruth, y_pred=binary_pred)
        f1 = f1_score(y_true=groundtruth, y_pred=binary_pred)
        auc_score = roc_auc_score(groundtruth, predicts)
        return accuracy, auc_score, prc, rc, f1


def evaluation_model(data, params):
    embedding_ftr, pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    batches = mini_batches_PNExtended(X_ftr=embedding_ftr, X_msg=pad_msg, X_added_code=pad_added_code, X_removed_code=pad_removed_code, 
                                        Y=labels, mini_batch_size=params.batch_size, shuffled=False)     

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    params.embedding_ftr = embedding_ftr.shape[1]

    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchNetExtended(args=params)    
    model.load_state_dict(torch.load(params.load_model))
    if torch.cuda.is_available():
        model = model.cuda()

    accuracy, roc_score, prc, rc, f1 = running_evaluation(model=model, data=batches)    
    print('Test data -- Accuracy: %.4f -- AUC: %.4f -- Precision: %.4f -- Recall: %.4f -- F1: %.4f' % (accuracy, roc_score, prc, rc, f1))    

