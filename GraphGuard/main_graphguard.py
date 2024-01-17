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

def train(args, model, device, train_graphs, optimizer, epoch, criterion, model_time):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = range(total_iters)

    loss_accum = 0
    for pos in pbar:
        # random.shuffle(train_graphs)
        # selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        # batch_graph = [train_graphs[idx] for idx in selected_idx]
        batch_graph = random.sample(train_graphs, args.batch_size)
        
        start = get_time()
        output = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        # compute loss
        loss = criterion(output, labels)
        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss
        
        duartion = get_time() - start
        model_time.append(duartion)

    average_loss = loss_accum / total_iters
    print(f"{epoch}, loss training: {average_loss}, {np.mean(model_time[1:])}")

    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

@torch.no_grad()
def test(args, model, device, train_graphs, test_graphs):
    model.eval()
    acc_train = 0.0
    # output = pass_data_iteratively(model, train_graphs)
    # pred = output.max(1, keepdim=True)[1]
    # labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    # correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    # acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
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
    parser.add_argument('--dataset', type=str, default="DBLP_v1",
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
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    
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
    parser.add_argument("--hash-method", type=str, default='md5', choices=["add", "hash", "md5", "sha256", "sha512", "sha1"], 
                        help="Hash function to use")
    parser.add_argument("--features-method", type=str, 
                        default="id", help="Feature function to use")
    parser.add_argument("--division-method", type=str, 
                        default="node", help="division measure to use")
    parser.add_argument('--degree_as_tag', type=int, default=0,
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument("--model_weight", type=str, default="model.pth",
                        help='model checkpoint path to load')
    parser.add_argument('--filename', type=str, default="output",
                        help='output file')
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
    print(111)
    graphs, num_classes, _ = load_dataset(args.dataset, args.degree_as_tag)
    train_graphs, test_graphs, _ = separate_data(graphs, args.seed, args.fold_idx, args.fold_n)

    print('#train_graphs:', len(train_graphs), '#test_graphs:', len(test_graphs))
    input_dim = train_graphs[0].node_features.shape[1]
    print('input dim:', input_dim)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, 
                        num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                        args.neighbor_pooling_type, device).to(device)
    # train model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    with open(args.filename, 'a+') as f:
        f.write('avg_loss,acc_train,acc_test\n')
    division_time = []
    start = get_time()
    if args.num_group <= 1 or args.division_func is None:
        print("Train without adversarial")
    else:
        print('processing...')
        train_graphs = sum([args.division_func(graph, args) for graph in tqdm(train_graphs)], start=[])
        test_graphs = sum([args.division_func(graph, args) for graph in tqdm(test_graphs)], start=[])
    division_time.append(get_time() - start)
    model_time = []
    print(111)
    best = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        avg_loss = train(args, model, device, train_graphs, optimizer, epoch, criterion, model_time)
        if epoch % 2 == 0:
            acc_train, acc_test = test(args, model, device, train_graphs, test_graphs)
            if acc_test>best:
                best = acc_test
                torch.save(model.state_dict(), args.model_weight)
            print(f"{epoch},{avg_loss},{acc_train},{acc_test}\n")
    # save model
    # torch.save(model.state_dict(), args.model_weight)
    print(f"{model_time=:}\n{division_time=:}", file=open(f"train_time_{args.dataset}.txt", "w"))
if __name__ == '__main__':
    main()