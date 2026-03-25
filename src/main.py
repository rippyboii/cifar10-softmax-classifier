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

def NormalizeData(X, mean_X, std_X):
    return (X - mean_X) / std_X

def InitNetwork(K, d, seed=42):
    rng = np.random.default_rng(seed)

    network = {}
    network["W"] = 0.01 * rng.standard_normal((K, d))
    network["b"] = np.zeros((K, 1))

    return network


def softmax(s):
    s_shifted = s - np.max(s, axis=0, keepdims=True)
    exp_s = np.exp(s_shifted)
    return exp_s / np.sum(exp_s, axis=0, keepdims=True)


def ApplyNetwork(X, network):
    W = network["W"]
    b = network["b"]

    s = W @ X + b
    P = softmax(s)

    return P

def ComputeLoss(P, y):
    n = P.shape[1]
    p_correct = P[y, np.arange(n)]
    L = -np.mean(np.log(p_correct))
    return L

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    data_dir = ROOT / "Datasets" / "cifar-10-python" / "cifar-10-batches-py"

    trainX, trainY, trainy = LoadBatch(data_dir / "data_batch_1")
    valX, valY, valy = LoadBatch(data_dir / "data_batch_2")
    testX, testY, testy = LoadBatch(data_dir / "test_batch")

    mean_X = np.mean(trainX, axis=1, keepdims=True)
    std_X = np.std(trainX, axis=1, keepdims=True)

    trainX = NormalizeData(trainX, mean_X, std_X)
    valX = NormalizeData(valX, mean_X, std_X)
    testX = NormalizeData(testX, mean_X, std_X)

    K = 10
    d = trainX.shape[0]

    net = InitNetwork(K, d)

    P = ApplyNetwork(trainX[:, 0:100], net)

    print("\nApplyNetwork check:")
    print("P shape:", P.shape)
    print("First column sum:", np.sum(P[:, 0]))
    print("Min prob:", np.min(P))
    print("Max prob:", np.max(P))

    print("\nNetwork shapes:")
    print("W shape:", net["W"].shape)
    print("b shape:", net["b"].shape)

    print("\nAfter normalization:")
    print("trainX overall mean:", np.mean(trainX))
    print("trainX overall std:", np.std(trainX))

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