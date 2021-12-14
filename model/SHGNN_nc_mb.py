import torch
import torch.nn as nn
import numpy as np
from ipdb import set_trace

from model.base_SHGNN import SHGNN_ctr_ntype_specific


class SHGNN_nc_mb_layer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 in_dim,
                 out_dim,
                 attn_drop=0.5):
        super(SHGNN_nc_mb_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim


        self.ctr_ntype_layer = SHGNN_ctr_ntype_specific(num_metapaths, etypes_list, in_dim)

        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        h = self.ctr_ntype_layer(inputs)
        h_fc = self.fc(h)
        return h_fc, h


class SHGNN_nc_mb(nn.Module):
    def __init__(self, num_metapaths, etypes_list, feats_dim_list, hidden_dim, out_dim, dropout_rate=0.5, adjD = None, count = None, count2= None):
        super(SHGNN_nc_mb, self).__init__()
        self.hidden_dim = hidden_dim
        self.adjD = adjD
        # node centrality encoding
        all_degree = list(set(count))
        all_degree.sort()
        self.degree_index = np.where(np.repeat(count[:,None], len(all_degree), axis=1) == np.repeat(np.array(all_degree)[None,:], adjD.shape[0], axis=0))[1]
        self.degree_embedding = torch.eye(len(all_degree))

        all_degree2 = list(set(count2))
        all_degree2.sort()
        self.degree_index2 = np.where(np.repeat(count2[:,None], len(all_degree2), axis=1) == np.repeat(np.array(all_degree2)[None,:], adjD.shape[0], axis=0))[1]
        self.degree_embedding2 = torch.eye(len(all_degree2))

        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        self.fc_list_c = nn.ModuleList([nn.Linear(len(all_degree)+len(all_degree2), hidden_dim, bias=True) for feats_dim in feats_dim_list])
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.7)

        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        for fc in self.fc_list_c:
            nn.init.xavier_normal_(fc.weight, gain=1.414)           

        # SHGNN_nc_mb layers
        self.layer1 = SHGNN_nc_mb_layer(num_metapaths, etypes_list, hidden_dim*2, out_dim,attn_drop=dropout_rate)

    def forward(self, inputs, target_node_indices):
        features_list, type_mask, adj_matrixes, feature_idxes = inputs
        device = features_list[0].device
        # type-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=device)
        transformed_features_c = torch.zeros(type_mask.shape[0], self.hidden_dim, device=device)
        
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = self.leaky_relu(fc(features_list[i]))
            transformed_features_c[node_indices] = self.leaky_relu(self.fc_list_c[i](torch.cat([self.degree_embedding[self.degree_index], self.degree_embedding2[self.degree_index2]], axis=1)[node_indices].to(device)))

        transformed_features = self.feat_drop(torch.cat([transformed_features, transformed_features_c], axis=1))
        transformed_features = transformed_features.half()

        logits, h = self.layer1((transformed_features, type_mask, adj_matrixes, feature_idxes))
        return logits[target_node_indices], h[target_node_indices]
