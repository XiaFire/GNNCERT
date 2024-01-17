from __future__ import print_function
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from math import log10
import argparse 
import math 


def multi_ci(counts, alpha):
    multi_list = []
    n = np.sum(counts)
    l = len(counts)
    for i in range(l): 
        multi_list.append(proportion_confint(min(max(counts[i], 1e-10), n-1e-10), n, alpha=alpha*2./l, method="beta"))
    return np.array(multi_list)

def BinoCP(nA,n, alpha):
    return proportion_confint(nA, n, alpha=2 * alpha, method="beta")[0]


def log_fact(n):
    return sum(log10(i) for i in range(1, n+1))


def nCr(d,delta, v):
    a_temp, b_temp, c_temp = log_fact(d), log_fact(v), log_fact(d-v)
    d_temp, e_temp, f_temp = log_fact(d-delta), log_fact(v), log_fact(d-delta-v)
    return 10**(d_temp-e_temp-f_temp-(a_temp-b_temp-c_temp))

def loop_search_compute_r(p_sn, p_ls, k_value, keep_value, dimension):
    if keep_value == 0:
        return np.inf
    
    p_sn = min(p_sn,1-p_ls)

    if p_ls <= p_sn/k_value:
        return -1 
    
    radius = 0
    while True:
        intersection_prob = nCr(dimension,radius,keep_value)
        intersection_prob_1 = nCr(dimension,radius+1,keep_value)
        if p_ls - (1 - intersection_prob) > (1- intersection_prob + p_sn)/k_value and p_ls - (1 - intersection_prob_1) <= (1- intersection_prob_1 + p_sn)/k_value:
            return radius
        radius += 1
    
def binary_search_compute_r(p_sn,p_ls,k_value,keep_value,dimension):
    if keep_value == 0:
        return np.inf
    
    p_sn = min(p_sn,1-p_ls)

    if p_ls <= p_sn/k_value:
        return -1 
    radius=0
    low, high =0, 1500
    while low <= high:
        radius = math.ceil((low+high)/2.0)
        intersection_prob = nCr(dimension,radius,keep_value)

        if p_ls - (1 - intersection_prob) > (1- intersection_prob + p_sn)/k_value:
            low = radius + 0.1 
        else:
            high = radius - 1
    radius = math.floor(low)
    intersection_prob = nCr(dimension,radius,keep_value)
    intersection_prob_1 = nCr(dimension,radius+1,keep_value)
    if p_ls - (1 - intersection_prob) > (1- intersection_prob + p_sn)/k_value and p_ls - (1 - intersection_prob_1) <= (1- intersection_prob_1 + p_sn)/k_value:
        return radius
    else:
        print("error", keep_value)
        return -2

def compute_radius(ls, probability_array, topk, keep_value,dimension):
    p_ls = probability_array[ls]
    probability_array[ls] = -1
    sorted_index = np.argsort(probability_array)[::-1]
    sorted_probability_topk = probability_array[sorted_index[0:topk]]
    p_sk = np.zeros([topk])
    radius_array = np.zeros([topk])
    for i in np.arange(sorted_probability_topk.shape[0]):
        p_sk[0:i+1] += sorted_probability_topk[i]
    for i in np.arange(topk):
        radius_array[i] = binary_search_compute_r(p_sk[i], p_ls, topk-i,keep_value,dimension)
    return np.amax(radius_array)


def compute_radius_call(vector_array,k=1,alpha=0.001,keep_value=1000,dimension=50176):
    class_freq = vector_array[:-1]
    CI = multi_ci(class_freq, alpha)
    pABar = CI[ls][0]
    probability_bar = CI[:,1]
    probability_bar = np.clip(probability_bar, a_min=-1, a_max=1-pABar)
    probability_bar[ls] = pABar
    r = compute_radius(ls, probability_bar,k, keep_value,dimension)
    return r 

if __name__ == "__main__":    

    input_file = './radii/'+args.dataset +'/'+args.keep+'/'+args.ns+'/'+'array_value.npz'
    data = np.load(input_file)['x']

    #print("Number of monte-carlo samples: %i"%(np.sum(data[idx,idx:data.shape[1]-1])))

    num_class = data.shape[1]-1
    num_data = data.shape[0] 
    #num_data = int(args.ts)
    certified_r = []

    #certified_radius_array = np.zeros([data.shape[0]],dtype = np.int)
    certified_radius_array = np.zeros([num_data],dtype = np.int)

    dst_filepath = './radii/'+args.dataset +'/'+args.keep+'/'+args.ns+'/'+'certified_radius_'+args.dataset+'_keep_'+args.keep+'_k_'+args.k+'_alpha_'+args.alpha+'_ns_'+args.ns+'_binocp.txt'
    dstnpz_filepath = './radii/'+args.dataset +'/'+args.keep+'/'+args.ns+'/'+'certified_radius_'+args.dataset+'_keep_'+args.keep+'_k_'+args.k+'_alpha_'+args.alpha+'_ns_'+args.ns+'_binocp.npz'

    if dst_filepath is not None:
        f = open(dst_filepath, 'w')
        print("idx\tradius", file=f, flush=True)

    for idx in range(num_data):
    #for idx in range(1):
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
        r = binary_search_compute_r(1-pAbar,pAbar, k_value=1,keep_value=int(args.keep),dimension= int(args.d))

        certified_radius_array[idx]=r
        print("{}\t{}".format(idx+1, r), flush=True)
        if dst_filepath is not None:
            print("{}\t{}".format(idx+1, r), file=f, flush=True)
    np.savez(dstnpz_filepath,x=certified_radius_array)
    if dst_filepath is not None:
        f.close()