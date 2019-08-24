import torch.nn as nn
import torch
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, args):
        super(PatchEmbedding, self).__init__()
        self.args = args

        V_code = args.vocab_code
        Dim = args.embedding_dim
        Class = args.class_num

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN-3D for commit code
        code_line = args.code_line  # the number of LOC in each hunk of commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_multiple_line = nn.ModuleList(
            [nn.Conv2d(Ci, Co, (K, self.args.num_filters * len(self.args.filter_sizes))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(len(Ks) * Co, args.hidden_units)  # hidden units
        self.fc2 = nn.Linear(args.hidden_units, Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):  # can be used for CNN-2D
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward_code(self, x, convs_line, convs_multiple_line):
        n_batch, n_hunk, n_line = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(n_batch * n_hunk * n_line, x.shape[3], x.shape[4])

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=convs_line)

        # apply cnn 2d for all lines in commit code
        x = x.reshape(n_batch, n_line, self.args.num_filters * len(self.args.filter_sizes))
        x = self.forward_msg(x=x, convs=convs_multiple_line)
        return x

    def forward(self, added_code, removed_code):
        # x_added_code = self.embed_code(added_code.cuda() if torch.cuda.is_available() else added_code)
        x_added_code = self.embed_code(added_code)
        x_added_code = self.forward_code(x_added_code, self.convs_code_line, self.convs_code_multiple_line)

        # x_removed_code = self.embed_code(removed_code.cuda() if torch.cuda.is_available() else removed_code)
        x_removed_code = self.embed_code(removed_code)
        x_removed_code = self.forward_code(x_removed_code, self.convs_code_line, self.convs_code_multiple_line)

        x_diff = x_added_code - x_removed_code  # measure the diff of the code changes

        x_commit = x_diff
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out

    def forward_commit_embeds(self, added_code, removed_code):
        x_added_code = self.embed_code(added_code.cuda() if torch.cuda.is_available() else added_code)
        x_added_code = self.forward_code(x_added_code, self.convs_code_line, self.convs_code_multiple_line)
        x_removed_code = self.embed_code(removed_code.cuda() if torch.cuda.is_available() else removed_code)
        x_removed_code = self.forward_code(x_removed_code, self.convs_code_line, self.convs_code_multiple_line)

        x_diff = x_added_code - x_removed_code  # measure the diff of the code changes

        x_commit = x_diff
        return x_commit
