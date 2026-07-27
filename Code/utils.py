import pickle
import os
import json
import math
import torch
import random
import hashlib
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from scipy.sparse import coo_matrix
from collections import defaultdict
from torch_scatter import scatter_sum
from torch_geometric.utils import softmax
from torch_geometric.data import Data, Batch
from lifelines.utils import concordance_index
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv, GATConv,GraphNorm,global_mean_pool
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
)

def load_adj_matrices(folder_path, drop_suffix=True):
    adj_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):  # 只处理csv文件
            file_path = os.path.join(folder_path, filename)
            adj_matrix = pd.read_csv(file_path, index_col=0)  # 如果第一列是索引
            #adj_matrix = pd.read_csv(file_path, header=None)  # 如果没有表头
            key = os.path.splitext(filename)[0] if drop_suffix else filename
            
            #adj_dict[key] = adj_matrix.values  # 存为 numpy 数组
            adj_dict[key] = adj_matrix       # 如果想保留 DataFrame
            
    return adj_dict

def build_pathway_dict(folder_path, cnv_amp_file, cnv_del_file, snv_file):
    
    cnv_amp = pd.read_csv(cnv_amp_file, index_col=0)
    cnv_del = pd.read_csv(cnv_del_file, index_col=0)
    snv = pd.read_csv(snv_file, index_col=0)

    pathway_dict = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue
        #print(filename)
        file_path = os.path.join(folder_path, filename)
        adj_matrix = pd.read_csv(file_path, index_col=0)
        genes_in_pathway = adj_matrix.index.tolist()

        sample_dict = {}
        for sample in cnv_amp.columns:  # 
            cnv_amp_vals = cnv_amp[sample].reindex(genes_in_pathway, fill_value=0)
            cnv_del_vals = cnv_del[sample].reindex(genes_in_pathway, fill_value=0) if sample in cnv_del.columns else pd.Series(0, index=genes_in_pathway)
            snv_vals = snv[sample].reindex(genes_in_pathway, fill_value=0) if sample in snv.columns else pd.Series(0, index=genes_in_pathway)
            df = pd.DataFrame({
                "CNV_amp": cnv_amp_vals,
                "CNV_del": cnv_del_vals,
                "SNV": snv_vals,
            })
            sample_dict[sample] = df
        key = os.path.splitext(filename)[0]
        pathway_dict[key] = sample_dict

    return pathway_dict

def build_pathway_dict_singleomic(folder_path, exp_file):

    exp_data = pd.read_csv(exp_file, index_col=0)

    pathway_dict = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue
        #print(filename)
        file_path = os.path.join(folder_path, filename)
        adj_matrix = pd.read_csv(file_path, index_col=0)
        genes_in_pathway = adj_matrix.index.tolist()

        sample_dict = {}
        for sample in exp_data.columns:  # 
            exp_vals = exp_data[sample].reindex(genes_in_pathway, fill_value=0)
            df = pd.DataFrame({
                "Exp": exp_vals
            })
            sample_dict[sample] = df
        key = os.path.splitext(filename)[0]
        pathway_dict[key] = sample_dict

    return pathway_dict

def dot_product_decode(Z):
    return torch.sigmoid(torch.mm(Z, Z.t()))

def seed_everything(seed = 3078):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_exp_result(setting, result, dir_path):
    ''' Save result dictionaries as JSON file'''
    exp_name = setting['exp_name']
    #del setting['max_epoch']
    #del setting['train_batch_size']
    #del setting['test_batch_size']

    hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    filename = dir_path+'/{}-{}.json'.format(exp_name, hash_key)
    result.update(setting)
    with open(filename, 'w') as f:
        json.dump(result, f)

def sparse_to_tuple(sparse_mx):
    if not isinstance(sparse_mx, coo_matrix):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).T  # (num_nonzero, 2)
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

class MultiHeadAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttentionModule, self).__init__()
        # 定义多头注意力层
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                                                    batch_first=True)
        # 定义层归一化和全连接层
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if isinstance(query, torch.sparse.Tensor):
            query = query.to_dense()

        if isinstance(key, torch.sparse.Tensor):
            key = key.to_dense()

        if isinstance(value, torch.sparse.Tensor):
            value = value.to_dense()

        # print(isinstance(value, torch.sparse.Tensor))

        # query, key, value 的维度应该是 [seq_len, batch_size, embed_dim]
        attn_output, attn_weights = self.multihead_attn(query, key, value, attn_mask=mask)
        # 跳跃连接 + 层归一化
        output = self.layer_norm(query + self.dropout(attn_output))
        # 输出经过全连接层
        output = self.fc(output)
        return output, attn_weights

def logging(msg, outdir, log_fpath):
    fpath = os.path.join(outdir, log_fpath)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    with open(fpath, 'a') as fw:
        fw.write("%s\n" % msg)
    print(msg)

def log_learning_rates(optimizer, outdir, filename='lr.log'):
    lr_info = " | ".join(
        f"Group {i}: {group['lr']:.4e}" 
        for i, group in enumerate(optimizer.param_groups[:-1])  # 不包含loss_fn组
    )
    logging(f'LR after update: {lr_info}', outdir, filename)

def build_sample_features(big_dict, pathway_order):
    
    sample_features = {}
    samples = list(next(iter(big_dict.values())).keys())

    for sample in samples:
        features = []
        for pathway in pathway_order:
            features.append(big_dict[pathway][sample])
        sample_features[sample] = torch.stack(features)
    return sample_features

def split_dict(sample_dict, ratios=(0.8, 0.1, 0.1), seed=666):
    keys = list(sample_dict.keys())
    random.seed(seed)
    random.shuffle(keys)  # 打乱顺序
    
    n = len(keys)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    
    train_keys = keys[:n_train]
    val_keys = keys[n_train:n_train + n_val]
    test_keys = keys[n_train + n_val:]
    
    train_dict = {k: sample_dict[k] for k in train_keys}
    val_dict = {k: sample_dict[k] for k in val_keys}
    test_dict = {k: sample_dict[k] for k in test_keys}
    
    return train_dict, val_dict, test_dict

def adj_df_to_edge_index(adj_df: pd.DataFrame):
    """把DataFrame邻接矩阵转成edge_index"""
    adj = adj_df.values
    row, col = np.nonzero(adj)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index, adj_df.index.tolist()  # 同时返回基因名顺序

class ExpressionDataset(Dataset):
    def __init__(self, expr_df: pd.DataFrame, labels: dict):
        self.expr_df = expr_df
        self.labels = labels
        self.sample_names = list(labels.keys())

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]
        label = self.labels[sample_name]
        return {
            'sample_name': sample_name,
            'label': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    sample_names = [item['sample_name'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    return {
        'sample_names': sample_names,
        'labels': labels
    }


  
# train function
def train(model,train_exp, train_loader, edge_index_b,edge_index_b_train, edge_weight_b,edge_weight_b_train,batchsize,loss_fn, optimizer):

    model.train()
    list_train_loss = []
    list_train_out = []
    list_train_true = []
    list_train_samples = []

    optimizer.zero_grad()

    for batch in tqdm(train_loader):
        sample_names = batch['sample_names']  # list[str]
        train_label = batch['labels'].cuda()

        sample_pred,_,_,_,_,_,_= model(train_exp, sample_names, edge_index_b,edge_index_b_train, edge_weight_b.unsqueeze(1),edge_weight_b_train.unsqueeze(1),batchsize)

        #y_true = torch.tensor(train_label[sample_name]).long().cuda()
        loss = loss_fn(sample_pred, train_label)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
        list_train_samples.append(sample_names)
        list_train_out.append(sample_pred.detach().cpu().numpy())
        list_train_loss.append(loss.detach().cpu().numpy())
        list_train_true.append(train_label.detach().cpu().numpy())

    return model, list_train_samples, list_train_out,list_train_loss, list_train_true

def validate(model,val_exp, val_loader, edge_index_b,edge_index_b_val, edge_weight_b,edge_weight_b_val,batchsize,loss_fn):
    model.eval()
    with torch.no_grad():  #####禁用梯度计算
    # ====== Test ====== #
        list_val_loss  = []
        list_val_out = []
        list_val_true = []
        list_val_sample = []
        
        for batch in tqdm(val_loader):
            sample_names = batch['sample_names']  # list[str]
            val_label = batch['labels'].cuda()
            
            sample_pred,_,_,_,_,_,_ = model(val_exp, sample_names, edge_index_b,edge_index_b_val, edge_weight_b.unsqueeze(1),edge_weight_b_val.unsqueeze(1),batchsize)
            output_loss = loss_fn(sample_pred, val_label)
            list_val_sample.append(sample_names)
            list_val_out.append(sample_pred.detach().cpu().numpy())
            list_val_loss.append(output_loss.detach().cpu().numpy())
            list_val_true.append(val_label.detach().cpu().numpy())
            
    return  list_val_sample, list_val_out,list_val_loss ,list_val_true

def test(model,test_exp, test_loader, edge_index_b,edge_index_b_test, edge_weight_b,edge_weight_b_test,batchsize,loss_fn):
    model.eval() 
    with torch.no_grad():  #####禁用梯度计算
        # ====== Test ====== #
        list_test_loss  = []
        list_test_out = []
        list_test_true = []
        list_test_sample = []

        for batch in tqdm(test_loader):
            sample_names = batch['sample_names']  # list[str]
            test_label = batch['labels'].cuda()
            
            sample_pred,_,_,_,_,_,_ = model(test_exp, sample_names, edge_index_b,edge_index_b_test, edge_weight_b.unsqueeze(1),edge_weight_b_test.unsqueeze(1),batchsize)
            output_loss = loss_fn(sample_pred, test_label)

            list_test_sample.append(sample_names)
            list_test_out.append(sample_pred.detach().cpu().numpy())
            list_test_loss.append(output_loss.detach().cpu().numpy())
            list_test_true.append(test_label.detach().cpu().numpy())
            
    return  list_test_sample,list_test_out,list_test_loss,list_test_true
