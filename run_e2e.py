import sys
import os
import time
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def setrandomseed(i):
    np.random.seed(seeds[i - 1])
    print(f"Seed: {seeds[i - 1]}, Random Number: {np.random.rand()}")
    random.seed(seeds[i - 1])
    print(f"Seed: {seeds[i - 1]}, Random Number: {random.random()}")


def main(method="cos", threshold=0.9, data="twitter15"):
    from E2E.conbinmodel import train
    from E2E.conbinmodel_test import test

    for i in range(1,11):
     #i=10
      setrandomseed(i)
      seed_val = seeds[i - 1]
      i = str(int(i))
      print(f"Running E2E pipeline with method: {method}, data: {data}")
      from pipeline_config import write_timing

      e2e_start = time.time()
      train(i, data, method=method, threshold=threshold, seed=seed_val)
      e2e_end = time.time()
      write_timing(
          seed=seed_val,
          dataset=data,
          method=method,
          pipeline="E2E",
          model="total",
          time_seconds=e2e_end - e2e_start,
      )
      test(i, data, method=method, threshold=threshold)


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "cos"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    data = sys.argv[3] if len(sys.argv) > 3 else "twitter15"
    main(method=method, threshold=threshold, data=data)
