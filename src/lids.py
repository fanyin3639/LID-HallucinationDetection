import os
import random
import torch
import faiss
import ast
import numpy as np
import pandas as pd
import torch.nn as nn
from metrics import roc
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, normalize
from torch.utils.data import Dataset, DataLoader

def compute_lid(y, sampled_feats, sample_size=-1, k_list=[200]):
    """
    Code borrowed and modified from this repo:
    https://github.com/TideDancer/iclr21_isotropy_contxt
    """

    cpu_index = faiss.IndexFlatL2(sampled_feats.shape[1])
    cpu_index.add(np.ascontiguousarray(sampled_feats))

    avg_lids = []

    for k in k_list:
        i = 0
        D = []
        b, nid = cpu_index.search(y, k)
        b = np.sqrt(b)
        D.append(b)

        D = np.vstack(D)
        rk = np.max(D, axis=1)
        rk[rk == 0] = 1e-8
        lids = D / rk[:, None]
        lids = -1 / np.mean(np.log(lids), axis=1)
        lids[np.isinf(lids)] = y.shape[1]  # if inf, set as space dimension
        lids = lids[~np.isnan(lids)]  # filter nan
        avg_lids.append(lids.tolist())
    avg_lids = np.array(avg_lids).mean(axis=0)
    return avg_lids


layers = [i for i in range(1, 25)]
p_value_for_layers = []

name = 'coqa'
name_map = {'tydiqa': 'Mistral-7B-v0.1_tydiqa', 'coqa': "Mistral-7B-v0.1_coqa", "xsum": "Llama-2-7b-hf_xsum"}



for i in layers:

    pd = torch.load(os.path.join('./output_tensors', f"{name_map[name]}_all_layer_{i}_pred.pt"))
    gt = torch.load(os.path.join('./output_tensors', f"{name_map[name]}_all_layer_{i}_gt.pt"))
    labels = torch.load(os.path.join('./output_tensors', f"{name_map[name]}_all_layer_1_label.pt"))

    pd = torch.stack(pd, dim=0)

    # choose the first 500 examples as test, set for testing
    test_idxs = [i for i in range(gt.shape[0]) if i < 500]
    train_idxs = [i for i in range(gt.shape[0]) if i not in test_idxs]

    train_pd = gt[train_idxs, :].numpy().astype('float32')
    train_labels = labels[train_idxs]


    test_pd = np.array(pd[test_idxs, :]).astype('float32')
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
        lids = compute_lid(test_pd, correct_batch, sample_size=-1, k_list=[k], block=50000)
        auroc = roc(test_labels, -lids)



