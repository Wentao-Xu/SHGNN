import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import datetime
from utils.pytorchtools import EarlyStopping
from utils.data import load_DBLP_data
from utils.tools import evaluate_results_nc
from model import SHGNN_nc_mb
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Params
out_dim = 4
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
etypes_list = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]]

global_log_file = None
def pprint(*args):
    time = '['+str(datetime.datetime.now())[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)

def run_model_DBLP(hidden_dim, num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    edge_metapath_indices_list, features_list, adjD, type_mask, labels, train_val_test_idx = load_DBLP_data()
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    num_author = len(np.where(type_mask == 0)[0])
    num_paper = len(np.where(type_mask == 1)[0])
    num_term = len(np.where(type_mask == 2)[0])
    num_venue = len(np.where(type_mask == 3)[0])
    # edge_metapath_indices_lists = []
    # for i in range(1):
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
    # np.save("./data/DBLP_count.npy", count)
    # np.save("./data/DBLP_count2.npy", count2)
    count = np.load("./data/DBLP_count.npy")
    count2 = np.load("./data/DBLP_count2.npy")
    adj_AP = torch.from_numpy(adjD[:num_author, num_author: num_author + num_paper]).float().to(device)
    adj_PT = torch.from_numpy(adjD[num_author: num_author + num_paper, num_author + num_paper: num_author + num_paper +num_term]).float().to(device)
    adj_PV = torch.from_numpy(adjD[num_author: num_author + num_paper, num_author + num_paper +num_term : len(type_mask)]).float().to(device)
    adj_ap = adj_AP / torch.sum(adj_AP, 0).repeat(num_author, 1)
    features_list[1] = torch.mm(adj_ap.t(), features_list[0])
    adj_pt = adj_PT / torch.sum(adj_PT, 0).repeat(num_paper, 1)
    features_list[2] = torch.mm(adj_pt.t(), features_list[1])
    adj_pv = adj_PV / torch.sum(adj_PV, 0).repeat(num_paper, 1)
    features_list[3] = torch.mm(adj_pv.t(), features_list[1])
    adj_AP = adj_AP.half()
    adj_PT = adj_PT.half()
    adj_PV = adj_PV.half()
    # for i in range(len(features_list)):
    #     features_list[i] = features_list[i].half()

    adj_matrixes = [[adj_AP, adj_AP.T], [adj_AP, adj_PT, adj_PT.T, adj_AP.T], [adj_AP, adj_PV, adj_PV.T, adj_AP.T]]
    feature_idxes = [
                    [[[0, 4057], [4057, 18385]], [[4057, 18385], [0, 4057]]],
                    [[[0, 4057], [4057, 18385]], [[4057, 18385], [18385, 26108]], [[18385, 26108], [4057, 18385]], [[4057, 18385], [0, 4057]]],
                    [[[0, 4057], [4057, 18385]], [[4057, 18385], [26108, 26128]], [[26108, 26128], [4057, 18385]], [[4057, 18385], [0, 4057]]]
                    ]

    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []
    for _ in range(repeat):
        net = SHGNN_nc_mb(3, etypes_list, [334, 334, 334, 334], hidden_dim, out_dim, dropout_rate, adjD, count, count2)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        target_node_indices = np.where(type_mask == 0)[0]

        for epoch in range(num_epochs):
            torch.cuda.empty_cache()
            t0 = time.time()
            # training
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
            # np.save("/data1/v-wentx/project/HGNN/SHGNN/visual/embedding/SHGNN_DBLP_embedding.npy", embeddings[test_idx].cpu().numpy())
            # np.save("/data1/v-wentx/project/HGNN/SHGNN/visual/embedding/SHGNN_DBLP_label.npy", labels[test_idx].cpu().numpy())
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
                embeddings[test_idx].cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim)
        svm_macro_f1_lists.append(svm_macro_f1_list)
        svm_micro_f1_lists.append(svm_micro_f1_list)
        nmi_mean_list.append(nmi_mean)
        nmi_std_list.append(nmi_std)
        ari_mean_list.append(ari_mean)
        ari_std_list.append(ari_std)

    # print out a summary of the evaluations
    svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2))
    svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2))
    nmi_mean_list = np.array(nmi_mean_list)
    nmi_std_list = np.array(nmi_std_list)
    ari_mean_list = np.array(ari_mean_list)
    ari_std_list = np.array(ari_std_list)
    global global_log_file
    # global_log_file = './output/DBLP/run.log'
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
    ap = argparse.ArgumentParser(description='testing for the DBLP dataset')
    ap.add_argument('--hidden-dim', type=int, default=128, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=20, help='Patience. Default is 20.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='DBLP', help='Postfix for the saved model and result. Default is DBLP.')

    args = ap.parse_args()
    run_model_DBLP(args.hidden_dim, args.epoch, args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)
