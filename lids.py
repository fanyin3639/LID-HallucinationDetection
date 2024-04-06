import os
import random
import torch
import faiss
import skdim
import ast
import numpy as np
import pandas as pd
import torch.nn as nn
import plotly.express as px
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, normalize
from torch.utils.data import Dataset, DataLoader
from skdim.id import TwoNN, ESS, MOM, KNN, DANCo, MiND_ML, MLE, lPCA
from sklearn import metrics


def compute_lid(y, sampled_feats, sample_size=-1, k_list=[200], metric='l2', block=50000):
    if metric == 'cos':
        cpu_index = faiss.IndexFlatIP(sampled_feats.shape[1])
        y = normalize(y)
        sampled_feats = normalize(sampled_feats)
    if metric == 'l2':
        cpu_index = faiss.IndexFlatL2(sampled_feats.shape[1])

    # print('cpu_index')
    # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    cpu_index.add(np.ascontiguousarray(sampled_feats))

    avg_lids = []

    for k in k_list:
        i = 0
        D = []
        while i < y.shape[0]:
           tmp = y[i:min(i + block, y.shape[0])]
           i += block
           b, nid = cpu_index.search(tmp, k)
           b = np.sqrt(b)
           D.append(b)

        D = np.vstack(D)
        # print("query finish")
        if metric == 'cos':
          D = 1 - D  # cosine dist = 1 - cosine
          D[D <= 0] = 1e-8
        rk = np.max(D, axis=1)
        rk[rk == 0] = 1e-8
        lids = D / rk[:, None]
        lids = -1 / np.mean(np.log(lids), axis=1)
        lids[np.isinf(lids)] = y.shape[1]  # if inf, set as space dimension
        lids = lids[~np.isnan(lids)]  # filter nan
        avg_lids.append(lids.tolist())
        # print('filter nan/inf shape', lids.shape)
        # print('k', k - 1, 'lid_mean', np.mean(lids), 'lid_std', np.std(lids))
    avg_lids = np.array(avg_lids).mean(axis=0)
    return avg_lids


def roc(corrects, scores):
    auroc = metrics.roc_auc_score(corrects, scores)
    return auroc


layers = [i for i in range(15, 22)]
p_value_for_layers = []

name = 'tydiqa'
name_map = {'tydiqa': 'Mistral-7B-v0.1_tydiqa', 'coqa': "Mistral-7B-v0.1_coqa", "xsum": "Llama-2-7b-hf_xsum"}



for i in layers:
    pd = torch.load(os.path.join('./output_tensors', f"{name_map[name]}_all_layer_{i}_pred.pt"))
    pds = []
    num_samples = []
    for k, v in pd.items():
        pds.append(v[:1, :])
        num_samples.append(v.shape[0])
    pd = torch.cat(pds)

    gt = torch.load(os.path.join('./output_tensors', f"{name_map[name]}_all_layer_{i}_gt.pt"))
    labels = torch.load(os.path.join('./output_tensors', f"{name_map[name]}_all_layer_1_label.pt"))


    true_pds = pd[labels == 1]
    wrong_pds = pd[labels == 0]

    # choose the first 500 examples as test
    test_idxs = [i for i in range(gt.shape[0]) if i < 500]
    train_idxs = [i for i in range(gt.shape[0]) if i not in test_idxs]

    train_pd = gt[train_idxs, :].numpy().astype('float32')
    train_labels = labels[train_idxs]


    test_pd = pd[test_idxs, :]
    test_gt = gt[test_idxs, :]
    test_labels = labels[test_idxs]

    correct_batch = []
    for p, l in zip(train_pd, train_labels):
        if l.item() == 1:
            correct_batch.append(p.tolist())
    correct_batch = np.array(correct_batch).astype('float32')


    numbers = correct_batch.shape[0]
    k_list = [numbers - 1]
    for k in k_list:
        lids = compute_lid(test_pd.numpy(), correct_batch, sample_size=-1, k_list=[k], metric='l2', block=50000)
        # gt_lids = compute_lid(test_gt.numpy(), correct_batch, sample_size=-1, k_list=[k], metric='l2', block=50000)
        auroc = roc(test_labels, -lids)



