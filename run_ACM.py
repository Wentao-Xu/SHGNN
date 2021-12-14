import time
import argparse

import torch.nn.functional as F
import torch.sparse
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_ACM_data
from utils.tools import evaluate_results_nc
from model import SHGNN_nc
import datetime

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Params
out_dim = 3
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
etypes_lists = [[0, 1], [2, 3]]
global_log_file = None
def pprint(*args):
    time = '['+str(datetime.datetime.now())[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


def run_model_ACM(num_layers, hidden_dim, num_epochs, patience, repeat, save_postfix):
    edge_metapath_indices_list, features_list, adjD, type_mask, labels, train_val_test_idx = load_ACM_data()
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    # feats_type is 2.
    num_paper = len(np.where(type_mask == 0)[0])
    num_author = len(np.where(type_mask == 1)[0])
    num_subject = len(np.where(type_mask == 2)[0])
    
    adj_PA = adjD[:num_paper, num_paper: num_paper + num_author]
    adj_PS = adjD[:num_paper, num_paper + num_author: len(type_mask)]
    adj_PA = torch.from_numpy(adj_PA).float().to(device)
    adj_PS = torch.from_numpy(adj_PS).float().to(device)
    # feature propataiton
    adj_md = adj_PA / torch.sum(adj_PA, 0).repeat(num_paper, 1)
    features_list[1] = torch.mm(adj_md.t(), features_list[0])
    adj_ma = adj_PS / torch.sum(adj_PS, 0).repeat(num_paper, 1)
    features_list[2] = torch.mm(adj_ma.t(), features_list[0])
    # edge_metapath_indices_lists = []
    # for i in range(len(edge_metapath_indices_list)):
    #     edge_metapath_indeces = []
    #     for j in edge_metapath_indices_list[i]:
    #         for k in edge_metapath_indices_list[i][j]:
    #             edge_metapath_indeces.append(k)
    #     edge_metapath_indices_lists.append(edge_metapath_indeces)
    # edge_metapath_indices_list = [torch.LongTensor(indices).to(device) for indices in edge_metapath_indices_lists]
    # count = np.zeros(adjD.shape[0])
    # count2 = np.zeros(adjD.shape[0])
    # for metapath in edge_metapath_indices_list:
    #     for k in tqdm(metapath):
    #         for i in metapath[k]:
    #             count[i[0]]+=1
    #             count2+=adjD[i[0]]
    #             count[i[1]]+=1
    #             count2+=adjD[i[1]]
    #             count[i[2]]+=1
    #             count2+=adjD[i[2]]
    # np.save("./data/ACM_count.npy", count)
    # np.save("./data/ACM_count2.npy", count2)
    count = np.load("./data/ACM_count.npy")
    count2 = np.load("./data/ACM_count2.npy")
    labels = torch.LongTensor(labels).to(device)
    feature_idxes = [
        [[[0, 4019], [4019, 11186]], [[4019, 11186],[0, 4019]]],
        [[[0, 4019], [11186, 11246]], [[11186, 11246],[0, 4019]]],
    ]

    adj_matrixes = [[adj_PA, adj_PA.T], [adj_PS, adj_PS.T]]

    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []

#Start Training
    for _ in range(repeat):
        net = SHGNN_nc(num_layers, 2, etypes_lists, [4000, 4000, 4000], hidden_dim, out_dim, dropout_rate, adjD, count, count2)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        target_node_indices = np.where(type_mask == 0)[0]

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        for epoch in range(num_epochs):
            t0 = time.time()

            # training forward
            net.train()
            logits, embeddings = net((features_list, type_mask, adj_matrixes, feature_idxes), target_node_indices)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp[train_idx], labels[train_idx])

            t1 = time.time()
            dur1.append(t1 - t0)

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t2 = time.time()
            dur2.append(t2 - t1)

            # validation forward
            net.eval()
            with torch.no_grad():
                logits, embeddings = net((features_list, type_mask, adj_matrixes, feature_idxes), target_node_indices)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])

            t3 = time.time()
            dur3.append(t3 - t2)

            # print info
            print(
                "Epoch {:05d} | Train_Loss {:.4f} | Val_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}".format(epoch, train_loss.item(), val_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break


        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        with torch.no_grad():
            logits, embeddings = net((features_list, type_mask, adj_matrixes, feature_idxes), target_node_indices)
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
                embeddings[test_idx].cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim)
        svm_macro_f1_lists.append(svm_macro_f1_list)
        svm_micro_f1_lists.append(svm_micro_f1_list)
        nmi_mean_list.append(nmi_mean)
        nmi_std_list.append(nmi_std)
        ari_mean_list.append(ari_mean)
        ari_std_list.append(ari_std)


#Evaluation
    svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2))
    svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2))
    nmi_mean_list = np.array(nmi_mean_list)
    nmi_std_list = np.array(nmi_std_list)
    ari_mean_list = np.array(ari_mean_list)
    ari_std_list = np.array(ari_std_list)
    global global_log_file
    # global_log_file = './output/ACM/run_acdataset.log'
    pprint('----------------------------------------------------------------')
    pprint('SVM tests summary')
    pprint('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
        zip(svm_macro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    pprint('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
        zip(svm_micro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    pprint('K-means tests summary')
    pprint('NMI: {:.6f}~{:.6f}'.format(nmi_mean_list.mean(), nmi_std_list.mean()))
    pprint('ARI: {:.6f}~{:.6f}'.format(ari_mean_list.mean(), ari_std_list.mean()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='testing for the ACM dataset')
    ap.add_argument('--layers', type=int, default=2, help='Number of layers. Default is 2.')
    ap.add_argument('--hidden-dim', type=int, default=128, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 10.')
    ap.add_argument('--repeat', type=int, default=10, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='ACM', help='Postfix for the saved model and result. Default is ACM.')

    args = ap.parse_args()
    run_model_ACM(args.layers, args.hidden_dim, args.epoch, args.patience, args.repeat, args.save_postfix)
