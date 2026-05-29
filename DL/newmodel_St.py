def nmodel(repeattimes, data, method="cos", threshold=0.4,seed=None):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import sys, os
    import time

    sys.path.append(os.getcwd())

    from dataset_config import DATASET_CONFIG
    from pipeline_config import (
        Method,
        CHECKPOINT_DIR,
        LOG_DIR,
        path_glan_train_outputs,
        path_glan_val_outputs,
        path_bigcn_train_outputs,
        path_bigcn_val_outputs,
        path_combined_model,
        write_timing,
    )

    from models.dl_combined import DLCombined

    cfg = DATASET_CONFIG[data]
    _method = Method.COS if method == "cos" else Method.THRED

    BiGCN_train_outputs = np.load(path_bigcn_train_outputs(_method, data, repeattimes))
    GLAN_train_outputs = np.load(path_glan_train_outputs(_method, data, repeattimes))
    y_train = np.load(os.path.join(CHECKPOINT_DIR, data + "_y_train.npy"))

    BiGCN_train_outputs_tensor = torch.from_numpy(BiGCN_train_outputs).float()
    GLAN_train_outputs_tensor = torch.from_numpy(GLAN_train_outputs).float()
    y_train_tensor = torch.from_numpy(y_train).long()

    BiGCN_val_outputs = np.load(path_bigcn_val_outputs(_method, data, repeattimes))
    GLAN_val_outputs = np.load(path_glan_val_outputs(_method, data, repeattimes))
    y_val = np.load(os.path.join(CHECKPOINT_DIR, data + "_y_dev.npy"))

    BiGCN_val_outputs_tensor = torch.from_numpy(BiGCN_val_outputs).float()
    GLAN_val_outputs_tensor = torch.from_numpy(GLAN_val_outputs).float()
    y_val_tensor = torch.from_numpy(y_val).long()

    train_dataset = TensorDataset(
        BiGCN_train_outputs_tensor, GLAN_train_outputs_tensor, y_train_tensor
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(
        BiGCN_val_outputs_tensor, GLAN_val_outputs_tensor, y_val_tensor
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    model = DLCombined(data)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 50
    log_path = os.path.join(LOG_DIR, f"DL_{data}_{method}_{threshold}_combined_training_{repeattimes}.txt")
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    model.train()
    with open(log_path, "w") as log_f:
        start_time = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct, total = 0, 0
            for batch_x1, batch_x2, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                #total_loss += loss.item()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
            
            train_loss = running_loss / len(train_loader)
            train_accuracy = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            with torch.no_grad():
                model.eval()
                val_outputs = model(BiGCN_val_outputs_tensor, GLAN_val_outputs_tensor)
                _, predicted = torch.max(val_outputs.data, 1)
                val_acc = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(
                    0
                )
                val_loss = criterion(val_outputs, y_val_tensor)
                model.train()

            log_f.write(
                f"Epoch [{epoch + 1}/{num_epochs}] train loss: {train_loss:.4f}  train acc: {train_accuracy * 100:.2f}%\n"
            )
            log_f.write(
               f"Epoch [{epoch + 1}/{num_epochs}] val loss: {val_loss:.4f}  val acc: {val_acc * 100:.2f}%"
            )
        end_time = time.time()
        write_timing(
            seed=seed if seed is not None else repeattimes,
            dataset=data,
            method=method,
            pipeline="DL",
            model="newmodel",
            time_seconds=end_time - start_time,)

    torch.save(model.state_dict(), path_combined_model(_method, data, repeattimes))
    
    from thop import profile
    from pipeline_config import write_flops

    model.eval()
    with torch.no_grad():
        _d1 = torch.randn(1, 600)
        _d2 = torch.randn(1, 256)
        f_dl, p_dl = profile(model, (_d1, _d2), verbose=False)
    write_flops(seed, data, method, "DL", "DLCombined", f_dl, p_dl)
    print(f"    DLCombined FLOPs={f_dl / 1e6:.2f}M  Params={p_dl / 1e6:.2f}M")
