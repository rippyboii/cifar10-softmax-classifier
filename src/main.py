import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def LoadBatch(filename):
    with open(filename, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    X = batch[b'data'].astype(np.float64) / 255.0   # (10000, 3072)
    X = X.T                                         # (3072, 10000)

    y = np.array(batch[b'labels'])                  # (10000,)
    K = 10
    n = X.shape[1]

    Y = np.zeros((K, n), dtype=np.float64)
    Y[y, np.arange(n)] = 1

    return X, Y, y


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    data_dir = ROOT / "Datasets" / "cifar-10-python" / "cifar-10-batches-py"

    trainX, trainY, trainy = LoadBatch(data_dir / "data_batch_1")
    valX, valY, valy = LoadBatch(data_dir / "data_batch_2")
    testX, testY, testy = LoadBatch(data_dir / "test_batch")

    print("trainX shape:", trainX.shape)
    print("trainY shape:", trainY.shape)
    print("trainy shape:", trainy.shape)

    print("valX shape:", valX.shape)
    print("valY shape:", valY.shape)
    print("valy shape:", valy.shape)

    print("testX shape:", testX.shape)
    print("testY shape:", testY.shape)
    print("testy shape:", testy.shape)

    print("X dtype:", trainX.dtype)
    print("X min/max:", trainX.min(), trainX.max())