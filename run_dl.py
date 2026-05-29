import sys
import os
import time
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
 

def setrandomseed(i):
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    np.random.seed(seeds[i - 1])
    print(f"Seed: {seeds[i - 1]}, Random Number: {np.random.rand()}")
    random.seed(seeds[i - 1])
    print(f"Seed: {seeds[i - 1]}, Random Number: {random.random()}")


def main(method="cos", threshold=0.4, data="twitter15"):
    from DL.generate_train_val_features import gen_tr
    from DL.generate_test_features import gen_te
    from DL.newmodel import nmodel
    from DL.newmodel_test import test
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    #seeds = [49, 50, 51]
    for i in range(1,11):
    # i=10
        setrandomseed(i)
        seed_val = seeds[i - 1]
        i = str(int(i))
        print(f"Running pipeline with method: {method}, data: {data}")
        from pipeline_config import write_timing
        gen_tr(i, data, method=method,threshold=threshold)
        gen_te(i, data, method=method)
        nmodel(i, data, method=method)
        test(i, data, method=method)


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "cos"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    data = sys.argv[3] if len(sys.argv) > 3 else "twitter15"
    main(method=method, threshold=threshold, data=data)
