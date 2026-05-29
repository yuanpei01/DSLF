def gen_te(repeattimes, data, method="cos", threshold=0.9):
    from process.prepared import read_dataset
    import os
    import pickle
    import torch
    from torch.utils.data import Dataset
    from tqdm import tqdm
    from process.prepared import loadTree, loadBiData, loadLabelData
    from torch.utils.data import DataLoader as TorchDataLoader, ConcatDataset
    from torch_geometric.data import DataLoader as GeoDataLoader, Batch
    import torch.optim as optim
    import torch.nn as nn
    from models.factories import create_glan, create_bigcn
    from itertools import zip_longest
    import numpy as np
    import os
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
    from sklearn.metrics import classification_report
    import sys, os

    sys.path.append(os.getcwd())

    from dataset_config import DATASET_CONFIG
    from pipeline_config import (
        Method,
        path_glan_best_model,
        path_bigcn_best_model,
        path_glan_log,
        path_bigcn_log,
        path_glan_test_outputs,
        path_bigcn_test_outputs,
    )

    cfg = DATASET_CONFIG[data]
    _method = Method.COS if method == "cos" else Method.THRED

    (
        X_train_tid,
        X_train_content,
        X_train_replies,
        y_train,
        X_dev_tid,
        X_dev_content,
        X_dev_replies,
        y_dev,
        X_test_tid,
        X_test_content,
        X_test_replies,
        y_test,
        train_data,
        val_data,
        test_data,
        train_labels,
        val_labels,
        test_labels,
        graph,
        word_embeddings,
        dic,
    ) = read_dataset(data, threshold=threshold)

    def cleandata(
        X_train_tid, X_train_content, X_train_replies, y_train, train_data, train_labels
    ):
        different_indices = [
            i for i in range(len(y_train)) if y_train[i] != train_labels[i]
        ]
        for index in sorted(different_indices, reverse=True):
            del X_train_tid[index]
            del X_train_content[index]
            del X_train_replies[index]
            del train_data[index]
            del y_train[index]
            del train_labels[index]
        return (
            X_train_tid,
            X_train_content,
            X_train_replies,
            y_train,
            train_data,
            train_labels,
        )

    X_train_tid, X_train_content, X_train_replies, y_train, train_data, train_labels = (
        cleandata(
            X_train_tid,
            X_train_content,
            X_train_replies,
            y_train,
            train_data,
            train_labels,
        )
    )
    print(y_train == train_labels)
    X_dev_tid, X_dev_content, X_dev_replies, y_dev, val_data, val_labels = cleandata(
        X_dev_tid, X_dev_content, X_dev_replies, y_dev, val_data, val_labels
    )
    print(y_dev == val_labels)
    X_test_tid, X_test_content, X_test_replies, y_test, test_data, test_labels = (
        cleandata(
            X_test_tid, X_test_content, X_test_replies, y_test, test_data, test_labels
        )
    )
    print(y_test == test_labels)
    print("X_train_tid:", len(X_train_tid))
    print("X_train_content:", len(X_train_content))
    print("X_train_replies:", len(X_train_replies))
    print("y_train:", len(y_train))
    print("train_data:", len(train_data))
    print("train_labels:", len(train_labels))

    class GLANDataset(Dataset):
        def __init__(self, tid, glandata, replies, labels):
            self.tid = tid
            self.glandata = glandata
            self.replies = replies
            self.labels = labels

            assert (
                len(self.tid)
                == len(self.glandata)
                == len(self.replies)
                == len(self.labels)
            ), "所有输入的长度必须一致"

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            try:
                tid = self.tid[idx]
                glandata = np.array(self.glandata[idx])
                replies = np.array(self.replies[idx])
                labels = self.labels[idx]
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                raise

            return {
                "tid": tid,
                "content": glandata,
                "replies": replies,
                "label": labels,
            }

    datasetname = cfg["display"]
    dataname = cfg["display"]
    TDdroprate = cfg["td_droprate"]
    BUdroprate = cfg["bu_droprate"]

    treeDic = loadTree(datasetname)

    print("X_train_tid 长度:", len(X_train_tid))
    print("X_train_content 长度:", len(X_train_content))
    print("X_train_replies 长度:", len(X_train_replies))
    print("y_train 长度:", len(y_train))

    BiGCN_train_dataset, BiGCN_test_dataset = loadBiData(
        dataname, treeDic, train_data, test_data, TDdroprate, BUdroprate
    )
    print("BiGCN_test_dataset:", len(BiGCN_test_dataset))
    BiGCN_test_loader = GeoDataLoader(
        BiGCN_test_dataset, batch_size=cfg["bigcn_batch_size"], shuffle=False
    )

    GLAN_test_dataset = GLANDataset(X_test_tid, X_test_content, X_test_replies, y_test)
    GLAN_test_loader = TorchDataLoader(
        GLAN_test_dataset, batch_size=cfg["glan_batch_size"], shuffle=False
    )

    config = {
        "lr": cfg["lr"],
        "reg": cfg["reg"],
        "batch_size": cfg["glan_batch_size"],
        "nb_filters": 100,
        "kernel_sizes": [3, 4, 5],
        "dropout": 0.5,
        "maxlen": 50,
        "epochs": cfg["glan_epochs"],
        "num_classes": cfg["num_classes"],
        "target_names": cfg["target_names"],
        "embedding_weights": word_embeddings,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelGLAN = create_glan(method, config, graph)

    import torch

    checkpoint = torch.load(path_glan_best_model(_method, data, repeattimes))
    modelGLAN.load_state_dict(checkpoint, strict=False)
    print("Model loaded successfully with modified state_dict.")

    modelGLAN = modelGLAN.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(modelGLAN.parameters(), lr=cfg["lr"])

    bestacc = 0
    num_epochs1 = 1
    for epoch in range(num_epochs1):
        modelGLAN.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        all_test_outputs = []
        for batch_gla in tqdm(GLAN_test_loader):
            X_test_tid = batch_gla["tid"]
            X_test_content = batch_gla["content"]
            X_test_replies = batch_gla["replies"]
            y_test = batch_gla["label"]

            X_test_tid = X_test_tid.to(device)
            X_test_content = X_test_content.to(device)
            X_test_replies = X_test_replies.to(device)
            y_test = y_test.to(device)

            optimizer.zero_grad()
            outputs, feature = modelGLAN(X_test_tid, X_test_content, X_test_replies)
            loss = criterion(outputs, y_test)

            total += y_test.size(0)
            correct += (outputs.argmax(dim=1) == y_test).sum().item()
            all_test_outputs.append(feature.detach())
        all_test_outputs = torch.cat(all_test_outputs, dim=0)
        np.save(
            path_glan_test_outputs(_method, data, repeattimes),
            all_test_outputs.cpu().numpy(),
        )
        print(f"Epoch [{epoch + 1}/{num_epochs1}],test Loss: {loss.item():.4f}")
        print(
            f"Epoch [{epoch + 1}/{num_epochs1}],test Accuracy: %.4f"
            % (100 * correct / total)
        )

        with open(path_glan_log(_method, data, repeattimes), "a+") as f:
            f.write(f"Epoch [{epoch + 1}/{num_epochs1}],test Loss: {loss.item():.4f}/n")
            f.write(
                f"Epoch [{epoch + 1}/{num_epochs1}],test Accuracy: %.4f"
                % (100 * correct / total)
            )

    lr2 = 0.0005
    weight_decay = 1e-4
    patience = 10
    in_feats = 5000
    hid_feats = 64
    out_feats = 64
    modelBiGCN = create_bigcn(data, in_feats, hid_feats, out_feats).to(device)
    modelBiGCN.load_state_dict(
        torch.load(path_bigcn_best_model(_method, data, repeattimes))
    )
    modelBiGCN = modelBiGCN.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(modelBiGCN.parameters(), lr=lr2)

    num_epochs2 = 1
    bestacc = 0
    for epoch in range(num_epochs2):
        modelBiGCN.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        all_test_outputs = []
        for batch_bi in tqdm(BiGCN_test_loader):
            inputs_bi = batch_bi
            labels_bi = batch_bi.y.to(device)
            print("inputs_bi 长度:", len(inputs_bi))
            inputs_bi = inputs_bi.to(device)

            optimizer.zero_grad()
            outputs, feature = modelBiGCN(inputs_bi)
            loss = criterion(outputs, labels_bi)

            print(labels_bi.type())

            total += labels_bi.size(0)
            correct += (outputs.argmax(dim=1) == labels_bi).sum().item()
            all_test_outputs.append(feature.detach())
        all_test_outputs = torch.cat(all_test_outputs, dim=0)
        np.save(
            path_bigcn_test_outputs(_method, data, repeattimes),
            all_test_outputs.cpu().numpy(),
        )
        print(f"Epoch [{epoch + 1}/{num_epochs2}],test Loss: {loss.item():.4f}")
        print(
            f"Epoch [{epoch + 1}/{num_epochs2}],test Accuracy: %.4f"
            % (100 * correct / total)
        )

        with open(path_bigcn_log(_method, data, repeattimes), "a+") as f:
            f.write(f"Epoch [{epoch + 1}/{num_epochs2}],test Loss: {loss.item():.4f}/n")
            f.write(
                f"Epoch [{epoch + 1}/{num_epochs2}],test Accuracy: %.4f"
                % (100 * correct / total)
            )
