import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import networkx as nx
from division import hash_func_map, features_func_map, division_func_map, get_node_id
from util import load_dataset, separate_data, get_time
from models.graphcnn import GraphCNN
import pickle
import copy

from tqdm import tqdm
from functools import partialmethod
from rich.progress import track

import torch_geometric.nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from torch_geometric.nn import GATConv, global_mean_pool
from util import DividedDataset, GAT, GCN


def train(args, model, device, train_loader, optimizer, epoch, criterion, model_time):
    model.train()

    loss_accum = 0
    for batch_data in train_loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        out = model(batch_data)
        loss = criterion(out, batch_data.y)
        loss.backward()
        loss_accum += loss.item()
        optimizer.step()
    average_loss = loss_accum
    print(f"{epoch}, loss training: {average_loss}")
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
@torch.no_grad()
def pass_data_iteratively(model, graphs, device, minibatch_size=64):
    model.eval()
    output = []
    
    dataset = DividedDataset(graphs)
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=False, num_workers=8)
    
    for batch_data in loader:
        batch_data = batch_data.to(device)
        out = model(batch_data)
        output.append(out)

    return torch.cat(output, 0)

@torch.no_grad()
def test(args, model, device, train_graphs, test_graphs):
    model.eval()

    output = pass_data_iteratively(model, train_graphs, device)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs, device)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--graphtype', type=str, default='ER',
                        help='type of graph generation')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                    help='number of iterations per each epoch')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    
    parser.add_argument('--fold_n', type=int, default=3,
                        help='the num of fold in n-fold validation.')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in n-fold validation. Should be less then n.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", default=False,
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument("--num_group", type=int, default=5,
                        help="Number of groups to divide the graph into")
    parser.add_argument("--hash-method", type=str, choices=["add", "hash", "md5", "sha256", "sha512", "sha1"], 
                        default="hash", help="Hash function to use")
    parser.add_argument("--features-method", type=str, 
                        default="features", help="Feature function to use")
    parser.add_argument("--division-method", type=str, 
                        default="features", help="division measure to use")
    parser.add_argument('--degree_as_tag', type=int,
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument("--model_weight", type=str,
                        help='model checkpoint path to load')
    parser.add_argument('--filename', type=str, default="output",
                        help='output file')
    parser.add_argument('--arch', type=str, default="GAT",
                        help='Graph Neural Network')
    args = parser.parse_args()
    args.features_func = features_func_map.get(args.features_method, get_node_id)
    args.hash_func = hash_func_map.get(args.hash_method, hash)
    args.division_func = division_func_map.get(args.division_method, None)
    args.degree_as_tag = bool(args.degree_as_tag)

    # set up seeds and gpu device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    graphs, num_classes, _ = load_dataset(args.dataset, args.degree_as_tag)
    train_graphs, test_graphs, _ = separate_data(graphs, args.seed, args.fold_idx, args.fold_n)

    print('#train_graphs:', len(train_graphs), '#test_graphs:', len(test_graphs))
    input_dim = train_graphs[0].node_features.shape[1]
    print('input dim:', input_dim)

    ARCH = {'GAT':GAT, 'GCN':GCN}[args.arch]
    model = ARCH(in_dim=input_dim, classes=num_classes).to(device)
    
    # train model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    with open(args.filename, 'a+') as f:
        f.write('avg_loss,acc_train,acc_test\n')

    if args.num_group <= 1:
        print("Train without adversarial")

    print('processing...')
    division_time = []
    start = get_time()
    train_graphs = sum([args.division_func(graph, args) for graph in tqdm(train_graphs)], start=[])
    division_time.append(get_time() - start)
    
    batch_size = 32
    dataset = DividedDataset(train_graphs)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model_time = []
    for epoch in track(range(1, args.epochs + 1)):
        avg_loss = train(args, model, device, train_loader, optimizer, epoch, criterion, model_time)
        if epoch % 100 == 0:
            acc_train, acc_test = test(args, model, device, train_graphs, test_graphs)
            with open(args.filename, 'a+') as f:
                f.write(f"{epoch},{avg_loss},{acc_train},{acc_test}\n")
    # save model
    torch.save(model.state_dict(), args.model_weight)
    print(f"{model_time=:}\n{division_time=:}", file=open(f"train_time_{args.dataset}.txt", "w"))
if __name__ == '__main__':
    main()