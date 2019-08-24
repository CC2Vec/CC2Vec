import torch.nn as nn
import torch
import torch.nn.functional as F


class PatchNetExtented(nn.Module):
    def __init__(self, args):
        super(PatchNetExtented, self).__init__()
        self.args = args

        V_msg = args.vocab_msg
        V_code = args.vocab_code
        Dim = args.embed_size
        Class = args.class_num

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # CNN-3D for commit code
        code_line = args.code_line  # the number of LOC in each hunk of commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_hunk = nn.ModuleList([nn.Conv3d(Ci, Co, (K, code_line, Co * len(Ks))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(3 * len(Ks) * Co + args.extended_ftr, args.hidden_size)  # hidden units
        self.fc2 = nn.Linear(args.hidden_size, Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward_code(self, x, convs_line, convs_hunks):
        n_batch, n_hunk, n_line = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(n_batch * n_hunk * n_line, x.shape[3], x.shape[4])

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=convs_line)

        # apply cnn 3d for each hunk in a commit code
        x = x.reshape(n_batch, n_hunk, n_line, self.args.num_filters * len(self.args.filter_sizes))
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3).squeeze(3) for conv in convs_hunks]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return x

    def forward_code_fast(self, x):
        n_batch, n_hunk, n_line = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(n_batch * n_hunk * n_line, x.shape[3], x.shape[4])

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=self.convs_code)
        x = x.reshape(n_batch, n_hunk, n_line, self.args.num_filters * len(self.args.filter_sizes))
        x = torch.mean(x, dim=2)  # hunk features
        x = torch.mean(x, dim=1)  # extract features
        return x

    def forward_code_fast_ver1(self, x):
        x = torch.mean(x, dim=3)  # line features
        x = torch.mean(x, dim=2)  # hunk features

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=self.convs_code)
        return x

    def forward(self, ftr, msg, added_code, removed_code):
        x_msg = self.embed_msg(msg)
        x_msg = self.forward_msg(x_msg, self.convs_msg)

        x_added_code = self.embed_code(added_code)
        x_added_code = self.forward_code_fast(x=x_added_code)
        # x_added_code = self.forward_code_fast_ver1(x=x_added_code)

        x_removed_code = self.embed_code(removed_code)
        x_removed_code = self.forward_code_fast(x=x_removed_code)
        # x_removed_code = self.forward_code_fast_ver1(x=x_removed_code)

        x_ftr = ftr.float()

        x_commit = torch.cat((x_ftr, x_msg, x_added_code, x_removed_code), 1)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out
