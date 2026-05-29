def gen_tr(repeattimes, data, method="cos", threshold=0.9, seed=None):
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
    from itertools import zip_longest
    import numpy as np
    from sklearn.metrics import classification_report
    import sys
    import time
    from dataset_config import DATASET_CONFIG
    from pipeline_config import (
        Method,
        CHECKPOINT_DIR,
        path_glan_train_outputs,
        path_glan_val_outputs,
        path_glan_log,
        path_glan_best_model,
        path_bigcn_train_outputs,
        path_bigcn_val_outputs,
        path_bigcn_log,
        path_bigcn_best_model,
        write_timing,
    )
    from models.factories import create_glan, create_bigcn, load_glan_checkpoint
    from process.countflops import compute_params_and_flops
    from process.countflops import count_flops
    from process.countflops import count_parameters

    cfg = DATASET_CONFIG[data]
    _method = Method.COS if method == "cos" else Method.THRED

    sys.path.append(os.getcwd())

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

    print("X_train_tid:", len(X_train_tid))
    print("X_train_content:", len(X_train_content))
    print("X_train_replies:", len(X_train_replies))
    print("y_train:", len(y_train))
    print("train_data:", len(train_data))
    print("train_labels:", len(train_labels))

    def cleandata(
        X_train_tid, X_train_content, X_train_replies, y_train, train_data, train_labels
    ):
        different_indices = [
            i for i in range(len(train_labels)) if y_train[i] != train_labels[i]
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

    np.save(os.path.join(CHECKPOINT_DIR, f"{data}_y_train.npy"), np.array(y_train))
    np.save(os.path.join(CHECKPOINT_DIR, f"{data}_y_dev.npy"), np.array(y_dev))
    np.save(os.path.join(CHECKPOINT_DIR, f"{data}_y_test.npy"), np.array(y_test))

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
        BiGCN_train_dataset, batch_size=cfg["bigcn_batch_size"], shuffle=False
    )
    BiGCN_val_loader = GeoDataLoader(
        BiGCN_val_dataset, batch_size=cfg["bigcn_batch_size"], shuffle=False
    )

    GLAN_train_dataset = GLANDataset(
        X_train_tid, X_train_content, X_train_replies, y_train
    )
    GLAN_val_dataset = GLANDataset(X_dev_tid, X_dev_content, X_dev_replies, y_dev)
    GLAN_train_loader = TorchDataLoader(
        GLAN_train_dataset, batch_size=cfg["glan_batch_size"], shuffle=False
    )
    GLAN_val_loader = TorchDataLoader(
        GLAN_val_dataset, batch_size=cfg["glan_batch_size"], shuffle=False
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
    modelGLAN = load_glan_checkpoint(modelGLAN, cfg["glan_checkpoint"])
    print("Model loaded successfully with modified state_dict.")

    modelGLAN = modelGLAN.to(device)
    
    # ── GLAN FLOPs ──
    #from process.countflops import compute_avg_batch_flops_GLAN
    from pipeline_config import write_flops
    #p_g_est, f_g_est= compute_avg_batch_flops_GLAN(modelGLAN, (128, 50, 300))
    input_size = (128, 50, 300)  
    p_g_est, f_g_est = compute_params_and_flops(modelGLAN, input_size)
    write_flops(seed, data, method, "DL", "GLAN", f_g_est, p_g_est)

    print(
        f"   GLAN FLOPs={f_g_est / 1e6:.2f}M  Params={p_g_est / 1e6:.2f}M" # [GPU:{device_id}]
    )
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(modelGLAN.parameters(), lr=cfg["lr"])

    print("BiGCN_train_loader:", len(BiGCN_train_loader))
    print("GLAN_train_loader:", len(GLAN_train_loader))
    GLAN_start_time = time.time()
    bestacc = 0
    num_epochs1 = cfg["glan_epochs"]
    for epoch in range(num_epochs1):
        modelGLAN.train()
        total_loss = 0.0
        total = 0
        correct = 0
        all_train_outputs = []
        for batch_gla in tqdm(GLAN_train_loader):
            X_train_tid = batch_gla["tid"]
            X_train_content = batch_gla["content"]
            X_train_replies = batch_gla["replies"]
            y_train = batch_gla["label"]

            X_train_tid = X_train_tid.to(device)
            X_train_content = X_train_content.to(device)
            X_train_replies = X_train_replies.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()
            outputs, feature = modelGLAN(X_train_tid, X_train_content, X_train_replies)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            total += y_train.size(0)
            correct += (outputs.argmax(dim=1) == y_train).sum().item()
            all_train_outputs.append(feature.detach())
        all_train_outputs = torch.cat(all_train_outputs, dim=0)
        np.save(
            path_glan_train_outputs(_method, data, repeattimes),
            all_train_outputs.cpu().numpy(),
        )
        print(f"Epoch [{epoch + 1}/{num_epochs1}],train Loss: {loss.item():.4f}\n")
        print(
            f"Epoch [{epoch + 1}/{num_epochs1}],train Accuracy: %.4f\n"
            % (100 * correct / total)
        )

        with open(path_glan_log(_method, data, repeattimes), "a+") as f:
            f.write(
                f"Epoch [{epoch + 1}/{num_epochs1}],train Loss: {loss.item():.4f}\n"
            )
            f.write(
                f"Epoch [{epoch + 1}/{num_epochs1}],train Accuracy: %.4f\n"
                % (100 * correct / total)
            )

        modelGLAN.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        all_val_outputs = []
        for batch_gla in tqdm(GLAN_val_loader):
            X_val_tid = batch_gla["tid"]
            X_val_content = batch_gla["content"]
            X_val_replies = batch_gla["replies"]
            y_val = batch_gla["label"]

            X_val_tid = X_val_tid.to(device)
            X_val_content = X_val_content.to(device)
            X_val_replies = X_val_replies.to(device)
            y_val = y_val.to(device)

            optimizer.zero_grad()
            outputs, feature = modelGLAN(X_val_tid, X_val_content, X_val_replies)
            loss = criterion(outputs, y_val)

            total += y_val.size(0)
            correct += (outputs.argmax(dim=1) == y_val).sum().item()
            all_val_outputs.append(feature.detach())
        all_val_outputs = torch.cat(all_val_outputs, dim=0)
        np.save(
            path_glan_val_outputs(_method, data, repeattimes),
            all_val_outputs.cpu().numpy(),
        )
        print(f"Epoch [{epoch + 1}/{num_epochs1}],val Loss: {loss.item():.4f}")
        print(
            f"Epoch [{epoch + 1}/{num_epochs1}],val Accuracy: %.4f"
            % (100 * correct / total)
        )

        with open(path_glan_log(_method, data, repeattimes), "a+") as f:
            f.write(f"Epoch [{epoch + 1}/{num_epochs1}],val Loss: {loss.item():.4f}")
            f.write(
                f"Epoch [{epoch + 1}/{num_epochs1}],val Accuracy: %.4f\n"
                % (100 * correct / total)
            )

        if (100 * correct / total) >= bestacc:
            torch.save(
                modelGLAN.state_dict(),
                path_glan_best_model(_method, data, repeattimes),
            )
            print("save model")
            bestacc = 100 * correct / total
            print("save GLAN model")

    GLAN_end_time = time.time()
    with open(path_glan_log(_method, data, repeattimes), "a+") as f:
        f.write(
            f"------------time cost" + ":" + str(GLAN_end_time - GLAN_start_time) + "\n"
        )
    write_timing(
        seed=seed if seed is not None else repeattimes,
        dataset=data,
        method=method,
        pipeline="DL",
        model="GLAN",
        time_seconds=GLAN_end_time - GLAN_start_time,
    )

    import torch

    lr2 = 0.0005
    weight_decay = 1e-4
    patience = 10
    in_feats = 5000
    hid_feats = 64
    out_feats = 64
    modelBiGCN = create_bigcn(data, in_feats, hid_feats, out_feats).to(device)
    modelBiGCN.load_state_dict(torch.load(cfg["bigcn_pretrained"]))
    modelBiGCN = modelBiGCN.to(device)
    # ── BiGCN FLOPs ──
    from process.countflops import compute_avg_batch_flops_BiGCN, count_parameters
    from pipeline_config import write_flops

    #_bigcn_sample = next(iter(BiGCN_train_loader)).to(device)
    # 直接传 model 和 data
    #f_b_est = compute_avg_batch_flops_BiGCN(
    #    modelBiGCN,
    #    BiGCN_train_dataset,
    #    batch_size=128 # 或 128
    #)
    #p_b_est = count_parameters(modelBiGCN)
    #p_b_est, f_b_est = count_params_and_flops_BiGCN(modelBiGCN, _bigcn_sample)


    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(modelBiGCN.parameters(), lr=lr2)

    BiGCN_start_time = time.time()
    num_epochs2 = cfg["bigcn_epochs"]
    print("BiGCN_train_loader:", len(BiGCN_train_loader))
    print("GLAN_train_loader:", len(GLAN_train_loader))
    bestacc = 0
    params=0
    flops=0
    for epoch in range(num_epochs2):
        modelBiGCN.train()
        total_loss = 0.0
        total = 0
        correct = 0
        all_train_outputs = []
        for batch_bi in tqdm(BiGCN_train_loader):
            inputs_bi = batch_bi
            labels_bi = batch_bi.y.to(device)
            print("inputs_bi 长度:", len(inputs_bi))
            inputs_bi = inputs_bi.to(device)

            optimizer.zero_grad()
            outputs, feature = modelBiGCN(inputs_bi)
            loss = criterion(outputs, labels_bi)
            loss.backward()
            optimizer.step()

            print(labels_bi.type())
            flops = flops+count_flops(modelBiGCN, inputs_bi)

            total += labels_bi.size(0)
            correct += (outputs.argmax(dim=1) == labels_bi).sum().item()
            all_train_outputs.append(feature.detach())

        all_train_outputs = torch.cat(all_train_outputs, dim=0)
        np.save(
            path_bigcn_train_outputs(_method, data, repeattimes),
            all_train_outputs.cpu().numpy(),
        )
        print(f"Epoch [{epoch + 1}/{num_epochs2}],train Loss: {loss.item():.4f}")
        print(
            f"Epoch [{epoch + 1}/{num_epochs2}],train Accuracy: %.4f"
            % (100 * correct / total)
        )

        with open(path_bigcn_log(_method, data, repeattimes), "a+") as f:
            f.write(f"Epoch [{epoch + 1}/{num_epochs2}],train Loss: {loss.item():.4f}")
            f.write(
                f"Epoch [{epoch + 1}/{num_epochs2}],train Accuracy: %.4f\n"
                % (100 * correct / total)
            )

        modelBiGCN.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        all_val_outputs = []
        for batch_bi in tqdm(BiGCN_val_loader):
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
            all_val_outputs.append(feature.detach())
        all_val_outputs = torch.cat(all_val_outputs, dim=0)
        np.save(
            path_bigcn_val_outputs(_method, data, repeattimes),
            all_val_outputs.cpu().numpy(),
        )
        print(f"Epoch [{epoch + 1}/{num_epochs2}],val Loss: {loss.item():.4f}")
        print(
            f"Epoch [{epoch + 1}/{num_epochs2}],val Accuracy: %.4f"
            % (100 * correct / total)
        )

        with open(path_bigcn_log(_method, data, repeattimes), "a+") as f:
            f.write(f"Epoch [{epoch + 1}/{num_epochs2}],val Loss: {loss.item():.4f}")
            f.write(
                f"Epoch [{epoch + 1}/{num_epochs2}],val Accuracy: %.4f\n"
                % (100 * correct / total)
            )

        if (100 * correct / total) >= bestacc:
            torch.save(
                modelBiGCN.state_dict(),
                path_bigcn_best_model(_method, data, repeattimes),
            )
            print("save model")
            bestacc = 100 * correct / total
            print("save BiGCN model")

    BiGCN_end_time = time.time()
    params = count_parameters(modelBiGCN) 
    flops= (flops)/len(BiGCN_train_loader)
    #
    write_flops(seed, data, method, "DL", "BiGCN", flops,params)
    print(
        f"   BiGCN FLOPs={flops / 1e6:.2f}M  Params={params / 1e6:.2f}M"     #[GPU:{device_id}]
    )

    with open(path_bigcn_log(_method, data, repeattimes), "a+") as f:
        f.write(
            f"------------time cost"
            + ":"
            + str(BiGCN_end_time - BiGCN_start_time)
            + "\n"
        )
    write_timing(
        seed=seed if seed is not None else repeattimes,
        dataset=data,
        method=method,
        pipeline="DL",
        model="BiGCN",
        time_seconds=BiGCN_end_time - BiGCN_start_time,
    )
