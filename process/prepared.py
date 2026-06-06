import random
import numpy as np
import torch
from torch_geometric.data import Data
import re
from .preprocess_similarty_ST import *
from Stream2.Process.dataset import BiGraphDataset
from .dataload import load_bigcn_data
import pickle
import os
import torch.nn as nn

from dataset_config import DATASET_CONFIG


def list_files_in_directory(directory):
    files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            files.append(filename[:-4])
    return files


def read_dataset(data, threshold=0.9):
    cfg = DATASET_CONFIG[data]
    root_path = cfg["dataset_dir"]
    file_name = data

    _process_dir = os.path.dirname(os.path.abspath(__file__))

    X_all_tids = []
    X_all_uids = []

    X_train_tid, X_train_content, X_train_replies, y_train = read_train_dev_test(
        root_path, file_name, ".train", X_all_tids
    )
    X_dev_tid, X_dev_content, X_dev_replies, y_dev = read_train_dev_test(
        root_path, file_name, ".dev", X_all_tids
    )
    X_test_tid, X_test_content, X_test_replies, y_test = read_train_dev_test(
        root_path, file_name, ".test", X_all_tids
    )

    w2v_path = os.path.join(_process_dir, os.path.basename(cfg["w2v_bin"]))
    A1, A2, A3 = w2v_feature_extract(
        root_path, file_name, w2v_path, threshold=threshold
    )
    _, X_train_content, X_train_replies, y_train, word_embeddings, _ = A1
    _, X_dev_content, X_dev_replies, y_dev = A2
    _, X_test_content, X_test_replies, y_test = A3

    directory1 = cfg["bigcn_graph_dir"]
    all_files = list_files_in_directory(directory1)

    not_in_all_files_train = [
        index for index, tid in enumerate(X_train_tid) if tid not in all_files
    ]
    for index in sorted(not_in_all_files_train, reverse=True):
        del X_train_tid[index]
        del X_train_content[index]
        del X_train_replies[index]
        del y_train[index]

    not_in_all_files_dev = [
        index for index, tid in enumerate(X_dev_tid) if tid not in all_files
    ]
    for index in sorted(not_in_all_files_dev, reverse=True):
        del X_dev_tid[index]
        del X_dev_content[index]
        del X_dev_replies[index]
        del y_dev[index]

    not_in_all_files_test = [
        index for index, tid in enumerate(X_test_tid) if tid not in all_files
    ]
    for index in sorted(not_in_all_files_test, reverse=True):
        del X_test_tid[index]
        del X_test_content[index]
        del X_test_replies[index]
        del y_test[index]

    X_all_tids = np.concatenate((X_train_tid, X_dev_tid, X_test_tid), axis=0)
    with open(root_path + file_name + "_graph.txt", "r", encoding="utf-8") as input:
        edge_index, edges_weight = [], []
        for line in input.readlines():
            tmp = line.strip().split()
            src = tmp[0]
            for dst_ids_ws in tmp[1:]:
                dst, w = dst_ids_ws.split(":")
                X_all_uids.append(dst)
                edge_index.append([dst, src])
                edges_weight.append(float(w))

    print(X_all_tids[:5])
    print(X_all_uids[:5])
    X_all_tids = np.array(X_all_tids)
    X_all_uids = np.array(X_all_uids)
    X_id = sorted(set(np.concatenate((X_all_tids, X_all_uids))))
    num_node = len(X_id)
    print(f"Number of nodes: {num_node}")

    X_id_dic = {id: i + 1 for i, id in enumerate(X_id)}

    edges_list = []
    for tup in edge_index:
        id_0 = X_id_dic.get(tup[0], None)
        id_1 = X_id_dic.get(tup[1], None)
        if id_0 is not None and id_1 is not None:
            edges_list.append([id_0, id_1])
    edges_list = torch.LongTensor(edges_list).t()
    edges_weight = torch.FloatTensor(edges_weight)

    data_graph = Data(edge_index=edges_list, edge_weight=edges_weight)

    bigcn_tid, bigcn_data, bigcn_labels = load_bigcn_data(
        cfg["bigcn_data_file"], cfg["bigcn_label_file"]
    )
    aligned_train_eid = []
    aligned_train_labels = []
    aligned_val_eid = []
    aligned_val_labels = []
    aligned_test_eid = []
    aligned_test_labels = []

    for tid in X_train_tid:
        if tid in bigcn_tid:
            aligned_train_eid.append(tid)
            aligned_train_labels.append(bigcn_labels[tid])

    for tid in X_dev_tid:
        if tid in bigcn_tid:
            aligned_val_eid.append(tid)
            aligned_val_labels.append(bigcn_labels[tid])

    for tid in X_test_tid:
        if tid in bigcn_tid:
            aligned_test_eid.append(tid)
            aligned_test_labels.append(bigcn_labels[tid])

    train_eid_mapped = [X_id_dic.get(eid, -1) for eid in aligned_train_eid]
    val_eid_mapped = [X_id_dic.get(eid, -1) for eid in aligned_val_eid]
    test_eid_mapped = [X_id_dic.get(eid, -1) for eid in aligned_test_eid]

    print("train_eid_mapped:", train_eid_mapped[:5])

    return (
        train_eid_mapped,
        X_train_content,
        X_train_replies,
        y_train,
        val_eid_mapped,
        X_dev_content,
        X_dev_replies,
        y_dev,
        test_eid_mapped,
        X_test_content,
        X_test_replies,
        y_test,
        aligned_train_eid,
        aligned_val_eid,
        aligned_test_eid,
        aligned_train_labels,
        aligned_val_labels,
        aligned_test_labels,
        data_graph,
        word_embeddings,
        X_id_dic,
    )


def loadLabelData(obj, seed=42):
    random.seed(seed)

    if "Twitter" in obj:
        labelPath = os.path.join(
            "./Stream2/", "data/" + obj + "/" + obj + "_label_All.txt"
        )
        labelset_nonR = ["news", "non-rumor"]
        labelset_f = ["false"]
        labelset_u = ["unverified"]
        labelset_t = ["true"]

        print("Loading tree label")
        NR, F, U, T = [], [], [], []
        l1 = l2 = l3 = l4 = 0
        labelDic = {}

        with open(labelPath, "r") as f:
            for line in f:
                line = line.rstrip()
                label, _, eid = line.split("\t")
                labelDic[eid] = label.lower()

                if label in labelset_nonR:
                    NR.append(eid)
                    l1 += 1
                if labelDic[eid] in labelset_f:
                    F.append(eid)
                    l2 += 1
                if labelDic[eid] in labelset_t:
                    T.append(eid)
                    l3 += 1
                if labelDic[eid] in labelset_u:
                    U.append(eid)
                    l4 += 1

        print(f"Total labels: {len(labelDic)}")
        print(f"Non-rumor: {l1}, False: {l2}, True: {l3}, Unverified: {l4}")

        all_ids = sorted(labelDic.keys())
        all_labels = [labelDic[eid] for eid in all_ids]

        return all_ids, all_labels

    return [], []


def loadTree(dataname, seed=42):
    random.seed(seed)
    treeDic = {}

    if "Twitter" in dataname:
        cfg = (
            DATASET_CONFIG[dataname.lower()]
            if dataname.lower() in DATASET_CONFIG
            else DATASET_CONFIG["twitter15"]
        )
        treePath = os.path.join(
            "./Stream2/", "data/" + dataname + "/data.TD_RvNN.vol_5000.txt"
        )
        print("reading Twitter tree")
        for line in open(treePath):
            line = line.rstrip()
            parts = line.split("\t")
            eid, indexP, indexC = parts[0], parts[1], int(parts[2])
            max_degree, maxL, Vec = int(parts[3]), int(parts[4]), parts[5]

            if eid not in treeDic:
                treeDic[eid] = {}

            treeDic[eid][indexC] = {
                "parent": indexP,
                "max_degree": max_degree,
                "maxL": maxL,
                "vec": Vec,
            }

        for eid in treeDic:
            treeDic[eid] = dict(sorted(treeDic[eid].items()))
        print("tree no:", len(treeDic))

    if dataname == "Weibo":
        treePath = os.path.join("./Stream2/", "data/Weibo/weibotree.txt")
        print("reading Weibo tree")
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC, Vec = (
                line.split("\t")[0],
                line.split("\t")[1],
                int(line.split("\t")[2]),
                line.split("\t")[3],
            )

            if eid not in treeDic:
                treeDic[eid] = {}

            treeDic[eid][indexC] = {"parent": indexP, "vec": Vec}

        for eid in treeDic:
            treeDic[eid] = dict(sorted(treeDic[eid].items()))
        print("tree no:", len(treeDic))

    return treeDic


def loadData(dataname, treeDic, fold_x_train, fold_x_test, droprate, seed=42):
    random.seed(seed)
    data_path = os.path.join("./Stream2/", "data", dataname + "graph")

    print("loading train set")
    print("origional train no:", len(fold_x_train))
    traindata_list = GraphDataset(
        fold_x_train, treeDic, droprate=droprate, data_path=data_path
    )
    print("train no:", len(traindata_list))

    print("loading test set")
    print("origional test no:", len(fold_x_test))
    testdata_list = GraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))

    return traindata_list, testdata_list


def loadUdData(dataname, treeDic, fold_x_train, fold_x_test, droprate, seed=42):
    random.seed(seed)
    data_path = os.path.join("./Stream2/", "data", dataname + "graph")

    print("loading train set")
    print("origional train no:", len(fold_x_train))
    traindata_list = UdGraphDataset(
        fold_x_train, treeDic, droprate=droprate, data_path=data_path
    )
    print("train no:", len(traindata_list))

    print("loading test set")
    print("origional test no:", len(fold_x_test))
    testdata_list = UdGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))

    return traindata_list, testdata_list


def loadBiData(
    dataname, treeDic, fold_x_train, fold_x_test, TDdroprate, BUdroprate, seed=42
):
    random.seed(seed)
    data_path = os.path.join("./Stream2/", "data", dataname + "graph")

    print("loading train set")
    print("origional train no:", len(fold_x_train))
    traindata_list = BiGraphDataset(
        fold_x_train,
        treeDic,
        tddroprate=TDdroprate,
        budroprate=BUdroprate,
        data_path=data_path,
    )
    print("train no:", len(traindata_list))
    print("train type:", type(traindata_list))

    print("loading test set")
    print("origional test no:", len(fold_x_test))
    testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    print("test type:", type(testdata_list))

    return traindata_list, testdata_list
