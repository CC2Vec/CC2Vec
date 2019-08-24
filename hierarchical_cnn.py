import torch.nn as nn
import torch
import torch.nn.functional as F


class HierarchicalCNN(nn.Module):
    def __init__(self, args):
        super(HierarchicalCNN, self).__init__()
        self.args = args

        V_code = args.vocab_code
        Dim = args.embed_size
        Class = args.class_num

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes
        self.batch_size = args.batch_size

        # standard neural network layer
        self.standard_nn_layer = nn.Linear(2 * (args.embed_size + Co), args.embed_size + Co)

        # neural network tensor
        self.W_nn_tensor_one = nn.Linear(args.embed_size + Co, args.embed_size + Co)
        self.W_nn_tensor_two = nn.Linear(args.embed_size + Co, args.embed_size + Co)

        # CNN-3D for commit code
        code_line = args.code_line  # the number of LOC in each hunk of commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_hunk = nn.ModuleList([nn.Conv3d(Ci, Co, (K, code_line, Co * len(Ks))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(3 * (args.embed_size + Co) + 3, args.hidden_size)  # hidden units
        self.fc2 = nn.Linear(args.hidden_size, Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):  # can be used for CNN-2D
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

    def forward(self, added_code, removed_code):
        x_added_code = self.embed_code(added_code)
        x_added_code = self.forward_code(x_added_code, self.convs_code_line, self.convs_code_hunk)

        x_removed_code = self.embed_code(removed_code)
        x_removed_code = self.forward_code(x_removed_code, self.convs_code_line, self.convs_code_hunk)

        subtract = self.subtraction(added_code=x_added_code, removed_code=x_removed_code)
        multiple = self.multiplication(added_code=x_added_code, removed_code=x_removed_code)
        # cos = self.cosine_similarity(added_code=x_added_code, removed_code=x_removed_code)
        euc = self.euclidean_similarity(added_code=x_added_code, removed_code=x_removed_code)
        nn = self.standard_neural_network_layer(added_code=x_added_code, removed_code=x_removed_code)
        ntn = self.neural_network_tensor_layer(added_code=x_added_code, removed_code=x_removed_code)

        x_diff = torch.cat((subtract, multiple, euc, nn, ntn), dim=1)
        x_diff = self.dropout(x_diff)

        x_commit = x_diff
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out

    def subtraction(self, added_code, removed_code):
        return added_code - removed_code

    def multiplication(self, added_code, removed_code):
        return added_code * removed_code

    def cosine_similarity(self, added_code, removed_code):
        cosine = nn.CosineSimilarity(eps=1e-6)
        return cosine(added_code, removed_code).view(self.batch_size, 1)

    def euclidean_similarity(self, added_code, removed_code):
        euclidean = nn.PairwiseDistance(p=2)
        return euclidean(added_code, removed_code).view(self.batch_size, 1)

    def standard_neural_network_layer(self, added_code, removed_code):
        concat = torch.cat((removed_code, added_code), dim=1)
        output = self.standard_nn_layer(concat)
        output = F.relu(output)
        return output

    def neural_network_tensor_layer(self, added_code, removed_code):
        output_one = self.W_nn_tensor_one(removed_code)
        output_one = torch.mul(output_one, added_code)
        output_one = torch.sum(output_one, dim=1).view(self.batch_size, 1)

        output_two = self.W_nn_tensor_two(removed_code)
        output_two = torch.mul(output_two, added_code)
        output_two = torch.sum(output_two, dim=1).view(self.batch_size, 1)

        W_output = torch.cat((output_one, output_two), dim=1)
        return F.relu(W_output)

    def forward_commit_embeds_diff(self, added_code, removed_code):
        x_added_code = self.embed_code(added_code.cuda() if torch.cuda.is_available() else added_code)
        x_added_code = self.forward_code(x_added_code, self.convs_code_line, self.convs_code_hunk)
        x_removed_code = self.embed_code(removed_code.cuda() if torch.cuda.is_available() else removed_code)
        x_removed_code = self.forward_code(x_removed_code, self.convs_code_line, self.convs_code_hunk)

        subtract = self.subtraction(added_code=x_added_code, removed_code=x_removed_code)
        multiple = self.multiplication(added_code=x_added_code, removed_code=x_removed_code)
        # cos = self.cosine_similarity(added_code=x_added_code, removed_code=x_removed_code)
        euc = self.euclidean_similarity(added_code=x_added_code, removed_code=x_removed_code)
        nn = self.standard_neural_network_layer(added_code=x_added_code, removed_code=x_removed_code)
        ntn = self.neural_network_tensor_layer(added_code=x_added_code, removed_code=x_removed_code)

        x_diff = torch.cat((subtract, multiple, euc, nn, ntn), dim=1)
        return x_diff

    def forward_commit_embeds(self, added_code, removed_code):
        x_added_code = self.embed_code(added_code.cuda() if torch.cuda.is_available() else added_code)
        x_added_code = self.forward_code(x_added_code, self.convs_code_line, self.convs_code_hunk)
        x_removed_code = self.embed_code(removed_code.cuda() if torch.cuda.is_available() else removed_code)
        x_removed_code = self.forward_code(x_removed_code, self.convs_code_line, self.convs_code_hunk)

        subtract = self.subtraction(added_code=x_added_code, removed_code=x_removed_code)
        multiple = self.multiplication(added_code=x_added_code, removed_code=x_removed_code)
        # cos = self.cosine_similarity(added_code=x_added_code, removed_code=x_removed_code)
        euc = self.euclidean_similarity(added_code=x_added_code, removed_code=x_removed_code)
        nn = self.standard_neural_network_layer(added_code=x_added_code, removed_code=x_removed_code)
        ntn = self.neural_network_tensor_layer(added_code=x_added_code, removed_code=x_removed_code)

        x_diff = torch.cat((subtract, multiple, euc, nn, ntn), dim=1)
        return x_diff
