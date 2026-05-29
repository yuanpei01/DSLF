def test(repeattimes, data, method="cos"):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import sys, os

    sys.path.append(os.getcwd())

    from dataset_config import DATASET_CONFIG
    from pipeline_config import (
        Method,
        CHECKPOINT_DIR,
        LOG_DIR,
        path_combined_model,
        path_glan_test_outputs,
        path_bigcn_test_outputs,
    )
    from models.dl_combined import DLCombined

    cfg = DATASET_CONFIG[data]
    _method = Method.COS if method == "cos" else Method.THRED

    BiGCN_test_outputs = np.load(path_bigcn_test_outputs(_method, data, repeattimes))
    GLAN_test_outputs = np.load(path_glan_test_outputs(_method, data, repeattimes))
    y_test = np.load(os.path.join(CHECKPOINT_DIR, data + "_y_test.npy"))

    BiGCN_test_outputs_tensor = torch.from_numpy(BiGCN_test_outputs).float()
    GLAN_test_outputs_tensor = torch.from_numpy(GLAN_test_outputs).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    print(BiGCN_test_outputs.shape)
    print(GLAN_test_outputs.shape)
    print(y_test.shape)

    test_dataset = TensorDataset(
        BiGCN_test_outputs_tensor, GLAN_test_outputs_tensor, y_test_tensor
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = DLCombined(data)
    model.load_state_dict(torch.load(path_combined_model(_method, data, repeattimes)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    from sklearn.metrics import classification_report

    with torch.no_grad():
        model.eval()
        test_outputs = model(BiGCN_test_outputs_tensor, GLAN_test_outputs_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f"test Accuracy: {accuracy * 100:.4f}%")
        target_names = cfg["label_names"]
        report = classification_report(
            y_test_tensor, predicted, target_names=target_names, digits=4
        )
        print(report)

    result_file = os.path.join(LOG_DIR, f"DL_test_results_{data}_{method}.txt")
    with open(result_file, "a+") as f:
        f.write(
            f"----------------------------------------"
            + repeattimes
            + "--------------------\n\n"
        )
        f.write(f"test Accuracy: {accuracy * 100:.4f}%\n\n")
        f.write(report)
