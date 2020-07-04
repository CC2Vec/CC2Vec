import torch.nn as nn
import torch
import torch.nn.functional as F


class PatchNetExtended(nn.Module):
    def __init__(self, args):
        super(PatchNetExtended, self).__init__()
        self.args = args

        V_msg = args.vocab_msg
        V_code = args.vocab_code
        Dim = args.embedding_dim
        Class = args.class_num
        Embedding = args.embedding_ftr

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)

        # CNN-3D for commit code
        code_line = args.code_line  # the number of LOC in each hunk of commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_hunk = nn.ModuleList([nn.Conv3d(Ci, Co, (K, code_line, Co * len(Ks))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(len(Ks) * Co + 2 * 2 * len(Ks) * Co + Embedding, args.hidden_units)  # hidden units
        self.fc2 = nn.Linear(args.hidden_units, Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward_code(self, x):        
        n_batch, n_file, n_hunk, n_line = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        new_batch = list()
        for b in range(0, n_batch):
            new_file = list()
            for f in range(0, n_file):
                new_hunk = list()
                for h in range(0, n_hunk):
                    line = x[b, f, h, :, :, :]
                    line = self.forward_msg(x=line, convs=self.convs_code_line)
                    new_hunk.append(line)
                new_file.append(torch.stack(new_hunk))            
            new_batch.append(torch.stack(new_file))        
        
        x = torch.stack(new_batch)

        new_batch = list()
        for b in range(0, n_batch):
            x_i = x[b, :, :, :, :]                        
            x_i = x_i.unsqueeze(1)  # (N, Ci, W, D)
            x_i = [F.relu(conv(x_i)).squeeze(3).squeeze(3) for conv in self.convs_code_hunk]
            x_i = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_i]            
            x_i = torch.cat(x_i, 1)
            x_i = torch.flatten(x_i)            
            new_batch.append(x_i)
        x = torch.stack(new_batch)
        return x

    def forward(self, ftr, msg, added_code, removed_code):
        x_msg = self.embed_msg(msg)
        x_msg = self.forward_msg(x_msg, self.convs_msg)        

        x_added_code = self.embed_code(added_code)
        x_added_code = self.forward_code(x=x_added_code)       

        x_removed_code = self.embed_code(removed_code)
        x_removed_code = self.forward_code(x=x_removed_code)        

        x_commit = torch.cat((ftr, x_msg, x_added_code, x_removed_code), 1)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out