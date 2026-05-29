def test(repead, data, method="cos", threshold=0.9):
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
    from models.e2e_combined import E2ECombined as CombinedModel
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
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from dataset_config import DATASET_CONFIG
    from pipeline_config import CHECKPOINT_DIR, LOG_DIR
    from models.factories import create_glan, create_bigcn, load_glan_checkpoint

    cfg = DATASET_CONFIG[data]

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

    from torch.utils.data import DataLoader as TorchDataLoader, ConcatDataset
    from torch_geometric.data import DataLoader as GeoDataLoader, Batch

    print("X_train_tid 长度:", len(X_train_tid))
    print("X_train_content 长度:", len(X_train_content))
    print("X_train_replies 长度:", len(X_train_replies))
    print("y_train 长度:", len(y_train))

    BiGCN_train_dataset, BiGCN_test_dataset = loadBiData(
        dataname, treeDic, train_data, test_data, TDdroprate, BUdroprate
    )
    print("BiGCN_test_dataset:", len(BiGCN_test_dataset))
    BiGCN_test_loader = GeoDataLoader(BiGCN_test_dataset, batch_size=128, shuffle=False)

    GLAN_test_dataset = GLANDataset(X_test_tid, X_test_content, X_test_replies, y_test)
    GLAN_test_loader = TorchDataLoader(GLAN_test_dataset, batch_size=128, shuffle=False)

    config = {
        "lr": cfg["lr"],
        "reg": cfg["reg"],
        "batch_size": 128,
        "nb_filters": 100,
        "kernel_sizes": [3, 4, 5],
        "dropout": 0.5,
        "maxlen": 50,
        "epochs": 30,
        "num_classes": cfg["num_classes"],
        "target_names": cfg["target_names"],
        "embedding_weights": word_embeddings,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_feats = 5000
    hid_feats = 64
    out_feats = 64

    modelGLAN = create_glan(method, config, graph)
    ##############################################################################
    lr2 = 0.0005
    weight_decay = 1e-4
    patience = 10
    in_feats = 5000
    hid_feats = 64
    out_feats = 64
    modelBiGCN = create_bigcn(data, in_feats, hid_feats, out_feats).to(device)
    ##############################################################################

    model = CombinedModel(modelGLAN, modelBiGCN, data).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #################################################################################
    model.load_state_dict(
        torch.load(
            os.path.join(
                CHECKPOINT_DIR, f"{data}_best{method}E2Emodel" + repead + ".pth"
            )
        )
    )
    ###################################################################################

    from sklearn.metrics import classification_report

    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    all_labels = []
    all_preds = []

    for batch_bi, batch_gla in tqdm(zip(BiGCN_test_loader, GLAN_test_loader)):
        inputs_bi = batch_bi
        labels_bi = batch_bi.y.to(device)
        print("inputs_bi 长度:", len(inputs_bi))
        inputs_bi = inputs_bi.to(device)
        X_test_tid = batch_gla["tid"]
        X_test_content = batch_gla["content"]
        X_test_replies = batch_gla["replies"]
        y_test = batch_gla["label"]

        X_test_tid = X_test_tid.to(device)
        X_test_content = X_test_content.to(device)
        X_test_replies = X_test_replies.to(device)
        y_test = y_test.to(device)

        with torch.no_grad():
            outputs = model(X_test_tid, X_test_content, X_test_replies, inputs_bi)

        loss = criterion(outputs, labels_bi)
        total_loss += loss.item()

        preds = outputs.argmax(dim=1)

        all_labels.extend(labels_bi.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        total += labels_bi.size(0)
        correct += (preds == labels_bi).sum().item()

    target_names = cfg["label_names"]
    report = classification_report(
        all_labels, all_preds, target_names=target_names, digits=4
    )
    accuracy = 100 * correct / total

    print(f" Test Loss: {total_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}%")
    print("Classification Report:\n", report)

    np.save(
        os.path.join(CHECKPOINT_DIR, f"{data}_E2E_{method}_all_labels.npy"), all_labels
    )
    np.save(
        os.path.join(CHECKPOINT_DIR, f"{data}_E2E_{method}_all_preds.npy"), all_preds
    )

    output_file = os.path.join(LOG_DIR, f"{data}_E2E_{method}_testresult.txt")
    with open(output_file, "a+") as f:
        f.write(f"------------------------------------" + repead + "\n")
        f.write(f"Test Loss: {total_loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}%\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
