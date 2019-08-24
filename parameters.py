import argparse


def read_args():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-train', action='store_true', help='training attention model')

    # Predicting our data
    parser.add_argument('--predict', action='store_true', help='predicting testing data')

    # Number of parameters for Attention model
    parser.add_argument('-embed_size', type=int, default=128, help='the dimension of embedding vector')
    parser.add_argument('-hidden_size', type=int, default=64, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training PatchNet')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('-log_interval', type=int, default=10,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=10,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500,
                        help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

    # Model
    parser.add_argument('-data_type', type=str, default='all', help='type of model for learning')
    parser.add_argument('-model', type=str, default='model', help='names of our model')

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')

    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')

    # option to load model
    parser.add_argument('-datetime', type=str, default=None, help='date of model [default: None]')
    parser.add_argument('-start_epoch', type=int, default=None, help='starting epoch of loading model')
    parser.add_argument('-end_epoch', type=int, default=None, help='ending epoch of loading model')
    parser.add_argument('-step', type=int, default=None, help='jumping step of the epoch')
    return parser


def read_args_cnn():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-train', action='store_true', help='training attention model')

    # Predicting our data
    parser.add_argument('--predict', action='store_true', help='predicting testing data')

    # Number of parameters for reformatting commits
    parser.add_argument('-msg_length', type=int, default=512, help='the length of the commit message')
    parser.add_argument('-code_hunk', type=int, default=8, help='the number of hunks in commit code')
    parser.add_argument('-code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=120, help='the length of each LOC of commit code')

    # Number of parameters for CNN model
    parser.add_argument('-embed_size', type=int, default=128, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_size', type=int, default=128, help='the number of nodes in hidden layers')

    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training PatchNet')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('-log_interval', type=int, default=10,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=10,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500,
                        help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

    # Model
    parser.add_argument('-data_type', type=str, default='all', help='type of model for learning')
    parser.add_argument('-model', type=str, default='model', help='names of our model')

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')

    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')

    # option to load model
    parser.add_argument('-datetime', type=str, default=None, help='date of model [default: None]')
    parser.add_argument('-start_epoch', type=int, default=None, help='starting epoch of loading model')
    parser.add_argument('-end_epoch', type=int, default=None, help='ending epoch of loading model')
    parser.add_argument('-step', type=int, default=None, help='jumping step of the epoch')
    return parser


def read_args_jiang():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-train', action='store_true', help='training PatchNet model')

    # Predicting our data
    parser.add_argument('--predict', action='store_true', help='predicting testing data')

    # Number of parameters for reformatting commits
    parser.add_argument('-msg_length', type=int, default=512, help='the length of the commit message')
    parser.add_argument('-code_hunk', type=int, default=8, help='the number of hunks in commit code')
    parser.add_argument('-code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=120, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('-embedding_dim', type=int, default=32, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=32, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=128, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training PatchNet')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('-log-interval', type=int, default=1,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=10,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500,
                        help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

    # Model
    parser.add_argument('-data_type', type=str, default='all', help='type of model for learning')
    parser.add_argument('-model', type=str, default='model', help='names of our model')

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')

    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')

    # option to load model
    parser.add_argument('-datetime', type=str, default=None, help='date of model [default: None]')
    parser.add_argument('-start_epoch', type=int, default=None, help='starting epoch of loading model')
    parser.add_argument('-end_epoch', type=int, default=None, help='ending epoch of loading model')
    return parser


def print_params(params):
    options = vars(params)
    for key in sorted(options.keys()):
        print(key, ':', options[key])
