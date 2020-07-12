# CC2Vec: Distributed Representations of Code Changes [[pdf](https://arxiv.org/pdf/2003.05620.pdf)]

## Contact
Questions and discussion are welcome: vdthoang.2016@smu.edu.sg

## Implementation Environment

Please install the neccessary libraries before running our tool:

- python==3.6.9
- torch==1.2.0
- tqdm==4.46.1
- nltk==3.4.5
- numpy==1.16.5
- scikit-learn==0.22.1

## Data & Pretrained models:

Please following the link below to download the data and pretrained models of our paper. 

- https://drive.google.com/file/d/1rPYGjw87YMNAdb2_2baO967i7ynp8sa2/view?usp=sharing

After downloading, simply copy the data and model folders to CC2Vec folder. 

## Hyperparameters:
We have a number of different parameters (Note that the number of hyperparameters is different depends on different tasks)

* --embedding_dim: Dimension of embedding vectors.
* --filter_sizes: Sizes of filters used by the hierarchical attention layers. 
* --num_filters: Number of filters. 
* --hidden_layers: Number of hidden layers. 
* --dropout_keep_prob: Dropout for training cc2vec. 
* --l2_reg_lambda: Regularization rate. 
* --learning_rate: Learning rate. 
* --batch_size: Batch size. 
* --num_epochs: Number of epochs. 

## Running and evalutation

### 1. Log message generation 

- In the first task (log message generation), simply run this command to train our model:

      $ python lmg_cc2ftr.py -train -train_data [path of our training data] -dictionary_data [path of our dictionary data]

- The command will create a folder snapshot used to save our model. To extract the code change features, please follow this command:

      $ python lmg_cc2ftr.py -predict -pred_data [path of our data] -dictionary_data [path of our dictionary data] -load_model [path of our model] -name [name of our output file]
      
- To evaluation the first task, please run this command:

      $ python lmg_eval.py -train_data [path of our training data] -test_data [path of our testing data] -train_cc2ftr_data [path of our code changes features extracted from training data] -test_cc2ftr_data [path of our code changes features extracted from testing data] 

### 2. Bug fixing patch identification

- Note that we need the training and testing dataset for this task. Please run this command to train our model:

      $ python bfp_cc2ftr.py -train -train_data [path of our training data] -test_data [path of our training data] -dictionary_data [path of our dictionary data]

- Similar to the first task, the command will create a folder snapshot used to save our model. To extract the code change features, please follow this command:

      $ python bfp_cc2ftr.py -predict -predict_data [path of our data] -dictionary_data [path of our dictionary data] -load_model [path of our model] -name [name of our output file]
      
- To train the model for bug fixing patch identification, please follow this command: 

      $ python bfp_PNExtended.py -train -train_data [path of our data] -train_data_cc2ftr [path of our code changes features extracted from training data] -dictionary_data [path of our dictionary data]
      
- To evaluate the model for bug fixing patch identification, please follow this command:
      
       $ python bfp_PNExtended.py -predict -pred_data [path of our data] -pred_data_cc2ftr [path of our code changes features extracted from our data] -dictionary_data [path of our dictionary data] -load_model [path of our model]
       
### 3. Just-in-time defect prediction

- For each dataset in just-in-time defect prediction (qt or openstack), we create two variants: one for training code changes features ('.pkl'), the other one for training just-in-time defect prediction model (end with '_dextend.pkl'). 

- Please run this command to train the code changes features:

      $ python jit_cc2ftr.py -train -train_data [path of our training data] -test_data [path of our training data] -dictionary_data [path of our dictionary data]

- Similar to the second task, the command will create a folder snapshot used to save our model. To extract the code change features, please follow this command:

      $ python jit_cc2ftr.py -predict -predict_data [path of our data] -dictionary_data [path of our dictionary data] -load_model [path of our model] -name [name of our output file]
      
- To train the model for just-in-time defect prediction, please follow this command: 

      $ python jit_DExtended.py -train -train_data [path of our data] -train_data_cc2ftr [path of our code changes features extracted from training data] -dictionary_data [path of our dictionary data]
      
- To evaluate the model for just-in-time defect prediction, please follow this command:
      
       $ python jit_DExtended.py -predict -pred_data [path of our data] -pred_data_cc2ftr [path of our code changes features extracted from our data] -dictionary_data [path of our dictionary data] -load_model [path of our model]

## Contact

Questions and discussion are welcome: vdthoang.2016@smu.edu.sg
