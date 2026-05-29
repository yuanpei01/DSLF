def train(repead, data, method="cos", threshold=0.9, seed=42):
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

    from dataset_config import DATASET_CONFIG
    from pipeline_config import CHECKPOINT_DIR, LOG_DIR, write_timing
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

    print("X_train_tid 长度:", len(X_train_tid))
    print("X_train_content 长度:", len(X_train_content))
    print("X_train_replies 长度:", len(X_train_replies))
    print("y_train 长度:", len(y_train))

    BiGCN_train_dataset, BiGCN_val_dataset = loadBiData(
        dataname, treeDic, train_data, val_data, TDdroprate, BUdroprate
    )
    print("BiGCN_train_dataset:", len(BiGCN_train_dataset))
    print("BiGCN_val_dataset:", len(BiGCN_val_dataset))
    BiGCN_train_loader = GeoDataLoader(
        BiGCN_train_dataset, batch_size=128, shuffle=False
    )
    BiGCN_val_loader = GeoDataLoader(BiGCN_val_dataset, batch_size=128, shuffle=False)
    for batch in BiGCN_train_loader:
        print(type(batch))
        print(batch)
        break
    GLAN_train_dataset = GLANDataset(
        X_train_tid, X_train_content, X_train_replies, y_train
    )
    GLAN_val_dataset = GLANDataset(X_dev_tid, X_dev_content, X_dev_replies, y_dev)
    GLAN_train_loader = TorchDataLoader(
        GLAN_train_dataset, batch_size=128, shuffle=False
    )
    GLAN_val_loader = TorchDataLoader(GLAN_val_dataset, batch_size=128, shuffle=False)

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
    modelGLAN = load_glan_checkpoint(modelGLAN, cfg["glan_checkpoint"])
    print("Model loaded successfully with modified state_dict.")

    # ── GLAN Params + FLOPs ─────────────────────────────────────────
    #from process.countflops import compute_avg_batch_flops_GLAN
    from pipeline_config import write_flops

    #p_g_est, f_g_est = compute_params_and_flops(modelGLAN, (128, 50, 300))
    #write_flops(seed, data, method, "E2E", "GLAN", f_g_est, p_g_est)
    #print(f"    GLAN FLOPs={f_g_est / 1e6:.2f}M  Params={p_g_est / 1e6:.2f}M")

    ##############################################################################
    lr2 = 0.0005
    weight_decay = 1e-4
    patience = 10
    in_feats = 5000
    hid_feats = 64
    out_feats = 64
    modelBiGCN = create_bigcn(data, in_feats, hid_feats, out_feats).to(device)
    modelBiGCN.load_state_dict(torch.load(cfg["bigcn_pretrained"], map_location="cpu"))

    # ── BiGCN Params + FLOPs ────────────────────────────────────────
    #from process.countflops import compute_avg_batch_flops_BiGCN, count_parameters
    #from pipeline_config import write_flops
    #_bigcn_sample = next(iter(BiGCN_train_loader)).to(device)
    #f_b_est = count_flops(modelBiGCN, _bigcn_sample) #* len(BiGCN_train_loader)
    #p_b_est = count_parameters(modelBiGCN)
    #f_b_est = compute_avg_batch_flops_BiGCN(
    #    modelBiGCN,
    #    BiGCN_train_dataset,
    #    batch_size=128 # 或 128
    #)
    #p_b_est = count_parameters(modelBiGCN)
    #write_flops(seed, data, method, "E2E", "BiGCN", f_b_est, p_b_est)
    #print(f"    BiGCN FLOPs={f_b_est / 1e6:.2f}M  Params={p_b_est / 1e6:.2f}M")

    ##############################################################################

    model = CombinedModel(modelGLAN, modelBiGCN, data).to(device)

    # ── E2ECombined Params + FLOPs ───────────────────────────────────
    from thop import profile

    model.eval()
    gla_sample = next(iter(GLAN_train_loader))
    bi_sample = next(iter(BiGCN_train_loader))
    with torch.no_grad():
        f_e2e, p_e2e = profile(
            model,
            (
                gla_sample["tid"].to(device),
                gla_sample["content"].to(device),
                gla_sample["replies"].to(device),
                bi_sample.to(device),
            ),
            verbose=False,
        )
    write_flops(
        seed=seed,
        dataset=data,
        method=method,
        pipeline="E2E",
        model="E2ECombined",
        flops=f_e2e,
        params=p_e2e,
    )
    print(f"    E2ECombined FLOPs={f_e2e / 1e6:.2f}M  Params={p_e2e / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    print("BiGCN_train_loader:", len(BiGCN_train_loader))
    print("GLAN_train_loader:", len(GLAN_train_loader))
    bestacc = 0

    #######################
    import time

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch_bi, batch_gla in tqdm(zip(BiGCN_train_loader, GLAN_train_loader)):
            inputs_bi = batch_bi
            labels_bi = batch_bi.y.to(device)
            print("inputs_bi 长度:", len(inputs_bi))
            inputs_bi = inputs_bi.to(device)
            X_train_tid = batch_gla["tid"]
            X_train_content = batch_gla["content"]
            X_train_replies = batch_gla["replies"]
            y_train = batch_gla["label"]

            X_train_tid = X_train_tid.to(device)
            X_train_content = X_train_content.to(device)
            X_train_replies = X_train_replies.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()
            outputs = model(X_train_tid, X_train_content, X_train_replies, inputs_bi)
            loss = criterion(outputs, labels_bi)
            loss.backward()
            optimizer.step()

            print(labels_bi.type())

            total += labels_bi.size(0)
            correct += (outputs.argmax(dim=1) == labels_bi).sum().item()

        print(f"Epoch [{epoch + 1}/{num_epochs}],train Loss: {loss.item():.4f}")
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],train Accuracy: %.4f"
            % (100 * correct / total)
        )

        with open(os.path.join(LOG_DIR, f"{data}_E2E_{method}_result.txt"), "a+") as f:
            f.write(f"------------------------------------" + repead + "/n")
            f.write(f"Epoch [{epoch + 1}/{num_epochs}],train Loss: {loss.item():.4f}/n")
            f.write(
                f"Epoch [{epoch + 1}/{num_epochs}],train Accuracy: %.4f"
                % (100 * correct / total)
            )

        model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_bi, batch_gla in tqdm(zip(BiGCN_val_loader, GLAN_val_loader)):
                inputs_bi = batch_bi
                labels_bi = batch_bi.y.to(device)
                print("inputs_bi 长度:", len(inputs_bi))
                inputs_bi = inputs_bi.to(device)
                X_val_tid = batch_gla["tid"]
                X_val_content = batch_gla["content"]
                X_val_replies = batch_gla["replies"]
                y_val = batch_gla["label"]

                X_val_tid = X_val_tid.to(device)
                X_val_content = X_val_content.to(device)
                X_val_replies = X_val_replies.to(device)
                y_val = y_val.to(device)

                outputs = model(X_val_tid, X_val_content, X_val_replies, inputs_bi)
                loss = criterion(outputs, labels_bi)

                print(labels_bi.type())

                total += labels_bi.size(0)
                correct += (outputs.argmax(dim=1) == labels_bi).sum().item()

        print(f"Epoch [{epoch + 1}/{num_epochs}],val Loss: {loss.item():.4f}")
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],val Accuracy: %.4f"
            % (100 * correct / total)
        )

        with open(os.path.join(LOG_DIR, f"{data}_E2E_{method}_result.txt"), "a+") as f:
            f.write(f"Epoch [{epoch + 1}/{num_epochs}],val Loss: {loss.item():.4f}/n")
            f.write(
                f"Epoch [{epoch + 1}/{num_epochs}],val Accuracy: %.4f"
                % (100 * correct / total)
            )
            f.close()

        if (100 * correct / total) >= bestacc:
            state_dict = model.state_dict()
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not k.endswith("total_ops") and not k.endswith("total_params")
            }
            torch.save(
                state_dict,
                os.path.join(
                    CHECKPOINT_DIR, f"{data}_best{method}E2Emodel" + repead + ".pth"
                ),
            )
            print("save model")
            bestacc = 100 * correct / total

#     end_time = time.time()
#     write_timing(
#         seed=seed,
#         dataset=data,
#         method=method,
#         pipeline="E2E",
#         model="E2Emodel",
#         time_seconds=end_time - start_time,
#     )
