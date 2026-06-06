# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:29:50 2024

@author: hp
"""

import torch
import torch.nn as th
from torch_geometric.nn import GCNConv

import copy

def scatter_mean(x, index, dim=0):
    # Get the unique indices
    unique_indices = torch.unique(index)
    out = torch.zeros((unique_indices.max() + 1, x.size(1)), device=x.device)
    
    # Sum the values per group (based on `index`) and count the number of occurrences
    for idx in unique_indices:
        mask = (index == idx)
        out[idx] = x[mask].mean(dim=dim)
    
    return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_flops(model, data):
    # Initialize FLOPs counter
    flops = 0
    batch_size = max(data.batch) + 1
    
    # Calculate FLOPs for TDrumorGCN
    in_feats = data.x.size(1)
    hid_feats = model.TDrumorGCN.conv1.out_channels
    out_feats = model.TDrumorGCN.conv2.out_channels
    E = data.edge_index.size(1)
    
    # FLOPs for conv1 in TDrumorGCN
    flops += 2 * E * in_feats * hid_feats
    
    # FLOPs for conv2 in TDrumorGCN
    flops += 2 * E * (hid_feats + in_feats) * out_feats
    
    # Calculate FLOPs for BUrumorGCN
    E = data.BU_edge_index.size(1)
    
    # FLOPs for conv1 in BUrumorGCN
    flops += 2 * E * in_feats * hid_feats
    
    # FLOPs for conv2 in BUrumorGCN
    flops += 2 * E * (hid_feats + in_feats) * out_feats
    
    # FLOPs for Linear layer in Net
    fc_in_feats = (out_feats + hid_feats) * 2
    flops += 2 * batch_size * fc_in_feats * 4  # For Linear
    
    return flops


# Assuming model and data are initialized
#from BiGCN.model.Twitter.BiGCN_Twitter import Net
#from BiGCN.model.Weibo.BiGCN_Weibo import Net
#in_feats=5000
#hid_feats=64
#out_feats=64
#model = Net(in_feats, hid_feats, out_feats)


# Dummy data for testing (replace with actual data)
#class Data:
#    x = torch.rand(1600, 50)  # 100 nodes with 16 features
#    edge_index = torch.randint(0, 1600, (2, 950))  # 200 edges
#    BU_edge_index = torch.randint(0, 1600, (2, 190))  # Same as above for BUrumorGCN
#    rootindex = torch.randint(0, 1600, (100,))  # Random root indices for batches
#    batch = torch.randint(0, 900, (100,))  # 10 different batches
    
#data = Data()

# Calculate parameters and FLOPs
#params = count_parameters(model)
#flops = count_flops(model, data)

#print(f"Number of parameters: {params}")
#print(f"Number of FLOPs: {flops}")

#with open('flops.txt','a+') as f:
#    f.write(f"BiGCNweibosimweight----------------------------------")
#    f.write(f"Number of parameters: {params}/n")
#    f.write(f"Number of FLOPs: {flops}/n")
    
    
####################################################################################
import torch
#from fvcore.nn import FlopCountAnalysis, parameter_count
from torch.autograd import Variable

# Import your GLAN model (assuming it's already defined as in your original code)
from Stream1.model.GGLAN.GLAN import GLAN

import torch
import torch.nn as nn
from torch.autograd import Variable
from Stream1.model.GGLAN.GAT import GATConv
def compute_params_and_flops(model, input_size):
    total_params = 0
    total_flops = 0
    
    # Helper function to calculate parameters
    def count_params(layer):
        return sum(p.numel() for p in layer.parameters())
    
    # Helper function to calculate FLOPs for a layer
    def count_flops_(layer, input_tensor):
        # For Linear layer: flops = input_features * output_features
        if isinstance(layer, nn.Linear):
            return input_tensor.size(0) * input_tensor.size(1) * layer.out_features
        
        # For Convolution layer: flops = output_width * output_height * kernel_size * in_channels * out_channels
        if isinstance(layer, nn.Conv1d):
            # Convolutional layer FLOPs calculation
            output_size = (input_tensor.size(2) - layer.kernel_size[0] + 1)  # Assuming stride=1, padding=0
            return output_size * input_tensor.size(0) * layer.in_channels * layer.out_channels * layer.kernel_size[0]
        
        # For GATConv layer: FLOPs can be computed based on attention mechanism and matrix multiplication
        if isinstance(layer, GATConv):
            num_nodes = input_tensor.size(0)
            in_features = layer.in_channels
            out_features = layer.out_channels
            heads = layer.heads
            # FLOPs for GAT (Matrix multiplication + attention mechanism)
            return (2 * in_features * out_features * heads) + (num_nodes * out_features * heads)
        
        return 0
    
    # Iterate through each layer of the model to compute params and flops
    for name, layer in model.named_modules():
        # Count parameters
        total_params += count_params(layer)
        
        # Example input tensor based on the size of the model's input
        if isinstance(layer, nn.Linear):
            input_tensor = torch.randn(input_size)
            total_flops += count_flops_(layer, input_tensor)
        elif isinstance(layer, nn.Conv1d):
            # Assuming input size is (batch_size, channels, length)
            input_tensor = torch.randn(input_size[0], input_size[1], input_size[2])
            total_flops += count_flops_(layer, input_tensor)
        elif isinstance(layer, GATConv):
            # For GAT layers, we assume that input_tensor is the node features
            input_tensor = torch.randn(input_size[0], input_size[1])  # Nodes, features
            total_flops += count_flops_(layer, input_tensor)
    
    return total_params, total_flops






# Instantiate the model
#model = GLAN(config, graph)

# Define input size: Example with batch_size=32, seq_length=100, word_dim=300
#print(word_embeddings.shape)
#input_size = (128, 50, 300)  # For example: 32 batches, 100 words in sequence, 300 embedding size

# Compute FLOPs and parameters
#params, flops = compute_params_and_flops(model, input_size)

# Print results
#print(f"Total parameters: {params}")
#print(f"Total FLOPs: {flops}")

#with open('flops2.txt','a+') as f:
#    f.write(f"GLANtwitter16fulltext----------------------------------")
#    f.write(f"Number of parameters: {params}/n")
#    f.write(f"Number of FLOPs: {flops}/n")




import torch
import torch.nn as th
from torch_geometric.nn import GCNConv

import copy
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def compute_avg_batch_flops_BiGCN(model, dataset, batch_size, num_classes=None):
    """
    返回 BiGCN 平均每个 batch 的 FLOPs，与 GLAN 的 per-batch FLOPs 直接可比。
    """
    in_feats = model.TDrumorGCN.conv1.in_channels
    hid_feats = model.TDrumorGCN.conv1.out_channels
    out_feats = model.TDrumorGCN.conv2.out_channels
    if num_classes is None:
        num_classes = model.fc.out_features

    total_N = 0          # 所有图的节点总数
    total_E_TD = 0       # 所有图的 TD 边数
    total_E_BU = 0       # 所有图的 BU 边数
    total_graphs = len(dataset)

    for data in dataset:
        total_N += data.x.size(0)
        total_E_TD += data.edge_index.size(1)
        total_E_BU += data.BU_edge_index.size(1)

    num_batches = (total_graphs + batch_size - 1) // batch_size

    # ========== TD 分支 ==========
    # TD conv1 + ReLU
    flops_td = 2 * total_N * in_feats * hid_feats   # 卷积乘法
    flops_td += 2 * total_E_TD * hid_feats          # 卷积加法（聚合）
    flops_td += total_N * hid_feats                 # ReLU

    # TD conv2（输入 dim = hid_feats）
    flops_td += 2 * total_N * hid_feats * out_feats
    flops_td += 2 * total_E_TD * out_feats
    flops_td += total_N * out_feats                 # ReLU

    # root_extend（TD侧，特征复制，FLOPs可忽略，加一点保底）
    flops_td += total_N * (in_feats + hid_feats)    # 复制操作估算

    # ========== BU 分支 ==========
    flops_bu = 2 * total_N * in_feats * hid_feats
    flops_bu += 2 * total_E_BU * hid_feats
    flops_bu += total_N * hid_feats

    flops_bu += 2 * total_N * hid_feats * out_feats
    flops_bu += 2 * total_E_BU * out_feats
    flops_bu += total_N * out_feats

    flops_bu += total_N * (in_feats + hid_feats)

    # ========== 融合部分 ==========
    feat_dim = hid_feats + out_feats

    # scatter_mean (TD + BU)  每个节点聚合一次，一次加法
    flops_fusion = total_N * feat_dim * 2           # 两个分支聚合
    flops_fusion += total_graphs * feat_dim * 2     # mean 除法+赋值

    # ========== 分类器 ==========
    feat_dim2 = (hid_feats + out_feats) * 2
    flops_fc = 2 * total_graphs * feat_dim2 * num_classes   # FC 乘加
    flops_fc += total_graphs * num_classes * 20              # softmax 估算

    total_flops = flops_td + flops_bu + flops_fusion + flops_fc
    avg_flops_per_batch = total_flops / num_batches

    return avg_flops_per_batch

