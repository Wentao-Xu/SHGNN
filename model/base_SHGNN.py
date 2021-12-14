import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax


def cal_cos_similarity(x, y): # the 2nd dimension of x and y are the same
    xy = x.mm(torch.t(y))
    x_norm =  torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
    y_norm =  torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
    cos_similarity = xy/(x_norm.mm(torch.t(y_norm)))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity

class SHGNN_metapath_specific(nn.Module):
    def __init__(self, etypes, out_dim):
        super(SHGNN_metapath_specific, self).__init__()
        self.out_dim = out_dim
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = edge_softmax

    def forward(self, inputs):
        features, type_mask, adj_matrix, feature_idx = inputs

        # Tree-attention Aggregator
        hidden = features[type_mask==0]
        device = hidden.device
        for i in range(len(adj_matrix)):
            child_features = features[feature_idx[i][0][0]: feature_idx[i][0][1],:]
            parent_features = features[feature_idx[i][1][0]: feature_idx[i][1][1],:]
            alpha = cal_cos_similarity(child_features, parent_features)
            alpha = torch.exp(alpha)
            alpha = alpha.mul(adj_matrix[i])
            alpha_sum = torch.sum(alpha, 0).reshape(1, -1).repeat(alpha.shape[0], 1)
            alpha_sum = alpha_sum.mul(adj_matrix[i])
            # alpha = alpha / (alpha_sum + 1e-6)
            alpha_sum = alpha_sum + (torch.ones(alpha.shape[0], alpha.shape[1], dtype=torch.float16).to(device) - adj_matrix[i])
            alpha = alpha / alpha_sum
            hidden = alpha.T.mm(hidden)

        ret = hidden
        return ret


class SHGNN_ctr_ntype_specific(nn.Module):
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 out_dim):
        super(SHGNN_ctr_ntype_specific, self).__init__()
        self.out_dim = out_dim

        self.metapath_layers = nn.ModuleList()

        for i in range(num_metapaths):
            self.metapath_layers.append(SHGNN_metapath_specific(etypes_list[i], out_dim))

        self.fc1 = nn.Linear(out_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, 1, bias=False)

        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        features, type_mask, adj_matrixes, feature_idxes = inputs

        metapath_outs = [F.elu(metapath_layer((features, type_mask, adj_matrix, feature_idx)).view(-1, self.out_dim)) for metapath_layer, adj_matrix, feature_idx in zip(self.metapath_layers, adj_matrixes, feature_idxes)]
        
        # meta-paths aggregator
        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out.to(torch.float32)))
            fc1_mean = torch.mean(fc1, dim=0) 
            fc2 = self.fc2(fc1_mean.to(torch.float32))
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0) 
        return h