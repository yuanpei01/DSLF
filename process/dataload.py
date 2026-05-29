def load_bigcn_data(data_file, label_file):
    data_dict = {}
    labels = {}
    data_list1 = []
    if "Twitter" in data_file:
        dic = {
            "non-rumor": 0,
            "false": 1,
            "unverified": 2,
            "true": 3,
        }
        with open(data_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                tid = parts[0]
                data_list1.append(tid)
                data = parts[1:]
                data_dict[tid] = data

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                tid = parts[2]
                label = parts[0]
                labels[tid] = dic[label]
    else:
        dic = {"0": 0, "1": 1}
        with open(data_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                tid = parts[0]
                data_list1.append(tid)
                data = parts[1:]
                data_dict[tid] = data

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                tid = parts[0]
                label = parts[1]
                labels[tid] = dic[label]

    return data_list1, data_dict, labels


def load_glan_data(train_file, dev_file, test_file):
    data_list = []
    labels = {}
    source = {}
    with open(train_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            tid = parts[0]
            source[tid] = parts[1:]
            data_list.append(tid)
            labels[tid] = parts[-1]

    with open(dev_file, "r") as g:
        for line in g:
            parts = line.strip().split()
            tid = parts[0]
            source[tid] = parts[1:]
            data_list.append(tid)
            labels[tid] = parts[-1]

    with open(test_file, "r") as h:
        for line in h:
            parts = line.strip().split()
            tid = parts[0]
            source[tid] = parts[1:]
            data_list.append(tid)
            labels[tid] = parts[-1]

    return data_list, labels, source
