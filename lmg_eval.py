import argparse
import pickle
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import re
from tqdm import tqdm

def finding_topK(cosine_sim, topK):
    cosine_sim = list(cosine_sim)
    topK_index = list()
    for i in range(topK):
        max_ = max(cosine_sim)
        index = cosine_sim.index(max_)
        topK_index.append(index)
        del cosine_sim[index]
    return topK_index

def get_data_index(data, indexes):
    return [data[i] for i in indexes]


def finding_bestK(diff_trains, diff_test, topK_index):
    if topK_index == None:        
        diff_code_train = [d.lower().split() for d in diff_trains]
    else:
        diff_code_train = get_data_index(data=diff_trains, indexes=topK_index)
        diff_code_train = [d.lower().split() for d in diff_code_train]

    diff_code_test = diff_test.lower().split()
    chencherry = SmoothingFunction()
    scores = [sentence_bleu(references=[diff_code_test], hypothesis=d, smoothing_function=chencherry.method1) for d in
            diff_code_train]
    bestK = scores.index(max(scores))
    
    return bestK

def clean_msg(messages):
    return [clean_each_line(line=msg) for msg in messages]

def clean_each_line(line):
    line = line.strip()
    line = line.split()
    line = ' '.join(line).strip()
    return line

def load_kNN_model(org_diff_code, tf_diff_code, ref_msg, topK=None):
    org_diff_train, org_diff_test = org_diff_code
    tf_diff_train, tf_diff_test = tf_diff_code
    ref_train, ref_test = ref_msg
    blue_scores = list()  

    run_tqdm = [i for i in range(tf_diff_test.shape[0])]  
    
    for i, (_) in enumerate(tqdm(run_tqdm)):   
        element = tf_diff_test[i, :]
        element = np.reshape(element, (1, element.shape[0]))
        cosine_sim = cosine_similarity(X=tf_diff_train, Y=element)

        if topK == None:
            bestK = finding_bestK(diff_trains=org_diff_train, diff_test=org_diff_test[i], topK_index=topK)
        else:
            topK_index = finding_topK(cosine_sim=cosine_sim, topK=topK)
            bestK = finding_bestK(diff_trains=org_diff_train, diff_test=org_diff_test[i], topK_index=topK_index)
        train_msg, test_msg = ref_train[bestK].lower(), ref_test[i].lower()        

        chencherry = SmoothingFunction()
        blue_score = sentence_bleu(references=[test_msg.split()], hypothesis=train_msg.split(), 
                                   smoothing_function=chencherry.method5)
        blue_scores.append(blue_score)    
    return blue_scores 

def read_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-train_data', type=str, default='./data/lmg/train.pkl', help='the directory of our training data')
    parser.add_argument('-test_data', type=str, default='./data/lmg/test.pkl', help='the directory of our training data')

    parser.add_argument('-train_cc2ftr_data', type=str, default='./data/lmg/train_cc2ftr.pkl', help='the directory of our training data')
    parser.add_argument('-test_cc2ftr_data', type=str, default='./data/lmg/test_cc2ftr.pkl', help='the directory of our training data')
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()
    data_train = pickle.load(open(params.train_data, "rb"))
    train_msg, train_diff = clean_msg(data_train[0]), data_train[1]

    data_test = pickle.load(open(params.test_data, "rb"))
    test_msg, test_diff = data_test[0], data_test[1]

    train_ftr = pickle.load(open(params.train_cc2ftr_data, "rb"))   
    test_ftr = pickle.load(open(params.test_cc2ftr_data, "rb"))

    org_diff_data = (train_diff, test_diff)
    tf_diff_data = (train_ftr, test_ftr)
    ref_data = (train_msg, test_msg)

    blue_scores = load_kNN_model(org_diff_code=org_diff_data, tf_diff_code=tf_diff_data, ref_msg=ref_data)
    print('Average of blue scores:', sum(blue_scores) / len(blue_scores) * 100)