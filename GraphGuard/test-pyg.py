import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from division import hash_func_map, features_func_map, get_node_tags, graph_structure_division, division_func_map
from util import load_data, separate_data, load_dataset, adversarial_train_test
from models.graphcnn import GraphCNN
from copy import deepcopy
import random
import networkx as nx
import itertools
import pandas as pd
import os
from evaluation import multi_ci, compute_radius, binary_search_compute_r, BinoCP

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rich.progress import track
from functools import partialmethod
from util import DividedDataset, GAT, GCN

def calculate_matric(data, args):
    args.alpha = 0.001

    # data 100 * (n_class+1)
    num_data = data.shape[0] 
    
    certified_radius_array = np.zeros([num_data],dtype = np.int)

    dst_filepath = os.path.join(args.dataset, '_binocp.txt')
    dstnpz_filepath = os.path.join(args.dataset, '_binocp.npz')


    if dst_filepath is not None:
        f = open(dst_filepath, 'w')
        print("idx\tradius", file=f, flush=True)

    for idx in range(num_data):
        ls = data[idx][-1]
        print("Number of monte-carlo samples: %i"%(np.sum(data[idx,0:data.shape[1]-1])))
        class_freq = data[idx][:-1]
        CI = multi_ci(class_freq, float(args.alpha))
        pABar = CI[ls][0]
        probability_bar = CI[:,1]
        probability_bar = np.clip(probability_bar, a_min=-1, a_max=1-pABar)
        probability_bar[ls] = pABar
        r = compute_radius(ls, probability_bar,int(args.k), int(args.keep),int(args.d))
        
        pAbar = BinoCP(class_freq[ls],sum(class_freq),float(args.alpha))
        r = binary_search_compute_r(1-pAbar,pAbar, k_value=1,keep_value=int(args.keep),dimension=int(args.d))

        certified_radius_array[idx]=r
        print("{},{}".format(idx+1, r), flush=True)
        if dst_filepath is not None:
            print("{},{}".format(idx+1, r), file=f, flush=True)

    np.savez(dstnpz_filepath,x=certified_radius_array)
    if dst_filepath is not None:
        f.close()

@torch.no_grad()
def pass_data_iteratively(model, graphs, device, minibatch_size=64):
    """
    Evaluate a model on a list of input graphs.

    Args:
        model: the model to be evaluated
        graphs: a list of input graphs
        minibatch_size: the size of each minibatch

    Returns:
        the concatenated output of the model for all input graphs
    """
    model.eval()
    output = []
    
    dataset = DividedDataset(graphs)
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=False, num_workers=8)
    
    for batch_data in loader:
        batch_data = batch_data.to(device)
        out = model(batch_data)
        output.append(out)

    return torch.cat(output, 0)



def merge_predictions(output, n):
    """
    Merge the predictions of multiple graphs by selecting the most frequent label.

    Args:
        output: a 1D numpy array of predicted labels
        n: the number of graphs

    Returns:
        a 1D numpy array of merged predictions
    """
    output = np.reshape(output, (-1, n))
    counts = np.apply_along_axis(np.bincount, axis=1, arr=output, minlength=n)
    merged_output = np.argmax(counts, axis=1)

    return merged_output

def get_Mp_pred(args, model, device, test_graphs, sub_func, **kwargs):
    """
    Compute the Mp score and predictions array for a given set of test graphs using a trained model.
        
    Args:
        args (Namespace): Namespace object containing the input arguments.
        model (nn.Module): Trained neural network model.
        device (torch.device): Device on which to run the computations.
        test_graphs (list): List of test graphs.
        sub_func (function): Function to use to divide the graph structures.
        **kwargs: Optional keyword arguments.

    Returns:
        Mp (numpy.ndarray): Array of Mp scores, one for each test graph.
        pred (numpy.ndarray): Array of predictions (True or False) indicating whether the predicted class matches the true class for each test graph.
    """
    n_initial = len(test_graphs) 

    labels = np.array([graph.label for graph in test_graphs])

    graphs = sum([sub_func(graph, args=args) for graph in test_graphs], start=[])
    
    n = len(graphs) // n_initial

    output = pass_data_iteratively(model, graphs, device, args.batch_size)
    output = output.detach().cpu().numpy()
    output = output.reshape(n_initial, n, -1)

    classes = output.shape[-1]
    out = output.argmax(-1)
    out = np.apply_along_axis(np.bincount, axis=1, arr=out, minlength=classes)

    # Compute I(l<c) and subtract it from out
    idx = out.argmax(axis=-1)
    _, cols = np.indices(out.shape)
    out[cols > idx[:, None]] -= 1
    # Sort the array
    out.sort(axis=-1)
    # Mp = {N_l - N_c + I(l < c)} / 2
    Mp = (out[:,-1] - out[:,-2]) / 2
    pred = idx == labels
    return Mp, pred

def get_Mp_RS(matrix, params, args):
    num_data = matrix.shape[0]
    Mp = np.zeros(num_data)
    for idx, (d, keep) in zip(range(num_data), params):
        ls = matrix[idx][-1]
        class_freq = matrix[idx][:-1]
        pAbar = BinoCP(class_freq[ls],sum(class_freq),float(args.alpha))
        r = binary_search_compute_r(1-pAbar,pAbar, k_value=1,keep_value=int(keep),dimension= int(d))
        Mp[idx]=r

    return Mp

def get_model_output(args, model, device, test_graphs, sub_func, **kwargs):
    '''
    Two alternative implementations for processing graphs using a minibatch approach.

    If the available computational resources are limited, 
    the first implementation is more suitable since it has a lower memory footprint.

    If the available computational resources are sufficient, 
    the second implementation is more suitable since it has a faster overall execution time.
    '''
    if args.use_second_implementation:
        # Implementation 1: Minibatch processing
        minibatch_size = args.minibatch_size
        outputs = []

        for i in track(range(0, len(test_graphs), minibatch_size), total=1+len(test_graphs)//minibatch_size):
            sample_graphs = test_graphs[i:i + minibatch_size]
            sample_graphs = sum([sub_func(graph, args=args, **kwargs) for graph in sample_graphs], start=[])
            output = pass_data_iteratively(model, sample_graphs, args.batch_size)
            output = output.detach().cpu().numpy()
            outputs.append(output)

        output = np.concatenate(outputs)

    else:
        # Implementation 2: Parallel processing
        from joblib import Parallel, delayed

        test_graphs = Parallel(n_jobs=-1)(
            delayed(sub_func)(graph, args=args, **kwargs)
            for graph in track(test_graphs, desc='Processing Graphs')
        )

        test_graphs = sum(test_graphs, start=[])
        output = pass_data_iteratively(model, test_graphs, device, args.batch_size)
        output = output.detach().cpu().numpy()

    return output

def get_pred_matrix_params(args, model, device, test_graphs, sub_func, **kwargs):
    model.eval()
    n_initial = len(test_graphs)
    n = args.num_group

    params = [len(graph.g) for graph in test_graphs]
    params = [n*(n-1)/2 for n in params]
    params = [(d, d*args.frac) for d in params]

    labels = np.array([graph.label for graph in test_graphs])

    output = get_model_output(args, model, device, test_graphs, sub_func, **kwargs)
    output = output.reshape(n_initial, n, -1)
    
    classes = output.shape[-1]
    output = output.argmax(axis=-1)

    output = np.apply_along_axis(np.bincount, axis=-1, arr=output, minlength=classes+1)
    idx = output.argmax(axis=-1)
    output[:, -1] = labels

    pred = idx == labels

    return pred, output, params

def test(args, model, device, test_graphs, sub_func, **kwargs):
    """
    Evaluate a model on a list of test graphs.

    Args:
        args: command line arguments
        model: the model to be evaluated
        device: the device to be used for evaluation
        test_graphs: a list of test graphs
        sub_func: a function to preprocess the input graphs
        **kwargs: additional keyword arguments for the sub_func

    Returns:
        the accuracy of testing sets
    """
    model.eval()
    n_initial = len(test_graphs)
    labels = np.array([graph.label for graph in test_graphs])

    test_graphs = sum([sub_func(graph, args=args, **kwargs) for graph in test_graphs], start=[])
    n = len(test_graphs) // n_initial
    
    output = pass_data_iteratively(model, test_graphs, device, args.batch_size)
    output = output.detach().cpu().numpy()
    output = output.argmax(axis=1)
    pred = merge_predictions(output, n)

    correct = np.sum(pred == labels)
    acc_test = correct / float(n_initial)

    return acc_test


def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch Graph Guard for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--graphtype', type=str, default='ER',
                        help='type of graph generation')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for evaluating (default: 512)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducing (default: 0)')
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
    parser.add_argument("--hash-method", type=str, choices=["hash", "md5", "sha256", "sha512"], 
                        default="hash", help="Hash function to use")
    parser.add_argument("--features-method", type=str, 
                        default="features", help="Feature function to use")
    parser.add_argument("--division-method", type=str, 
                        default="id", help="division measure to use")
    parser.add_argument('--degree_as_tag', type=int,
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument("--model_weight", type=str,
                        help='model checkpoint path to load')
    parser.add_argument("--defense_method", type=str, choices=['GraphGuard', 'GG', 'RS', 'all'], help="defense method to use")
    parser.add_argument('--use-second-implementation', action='store_false', help='Use the second implementation (minibatch processing)')
    parser.add_argument('--minibatch-size', type=int, default=10, help='Minibatch size for the first implementation')
    parser.add_argument('--randomly_preserve', type=float, default=0.1,
                    help='when adversarial_train_testing, randomly preserve certain fraction of entries')
    parser.add_argument('--frac', type=float, default=0.1,
                    help='when adversarial_train_testing, randomly preserve certain fraction of entries')
    parser.add_argument('--filename', type=str, default="output",
                        help='output file')
    parser.add_argument('--arch', type=str)
    args = parser.parse_args()
    args.features_func = features_func_map.get(args.features_method, get_node_tags)
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

    input_dim = train_graphs[0].node_features.shape[1]

    ARCH = {'GAT':GAT, 'GCN':GCN}[args.arch]
    model = ARCH(in_dim=input_dim, classes=num_classes).to(device)
    model_weights = torch.load(args.model_weight, map_location='cpu')
    model.load_state_dict(model_weights)

    sizes, accs = [], []

    if args.defense_method in ["GraphGuard", "GG"]:
        Mp, pred = get_Mp_pred(args, model, device, test_graphs, args.division_func)
        print(args.dataset, end=',')
        for size in range(int(Mp.max()+1)):
            sizes.append(size)
            acc = np.sum(pred & (Mp > size)) / len(Mp)
            accs.append(acc)
            print(acc, end=',')
        print('')
    elif args.defense_method == "RS":
        args.alpha = 0.001
        pred, matrix, params = get_pred_matrix_params(args, model, device, test_graphs, adversarial_train_test, tag2index=None, d=args.num_group)
        Mp = get_Mp_RS(matrix, params, args)
        for size in range(int(Mp[Mp!=np.inf].max()+1)):
            sizes.append(size)
            acc = np.sum(pred & (Mp >= size)) / len(Mp)
            accs.append(acc)
            print(size, acc)

    result = pd.DataFrame({"size":sizes, "acc":accs})
    result.to_csv(args.filename, index=False)

if __name__ == '__main__':
    main()