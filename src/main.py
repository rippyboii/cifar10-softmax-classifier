import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
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


def ComputeAccuracy(P, y):
    y_pred = np.argmax(P, axis=0)
    acc = np.mean(y_pred == y)
    return acc

def BackwardPass(X, Y, P, network, lam):
    W = network["W"]
    n = X.shape[1]

    G = P - Y

    grad_W = (G @ X.T) / n + 2 * lam * W
    grad_b = np.sum(G, axis=1, keepdims=True) / n

    grads = {
        "W": grad_W,
        "b": grad_b
    }

    return grads

def ComputeGradsWithTorch(X, y, network_params):

    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)

    # will be computing the gradient w.r.t. these parameters
    W = torch.tensor(network_params['W'], requires_grad=True)
    b = torch.tensor(network_params['b'], requires_grad=True)    
    
    N = X.shape[1]
    
    scores = torch.matmul(W, Xt)  + b;

    ## give an informative name to this torch class
    apply_softmax = torch.nn.Softmax(dim=0)

    # apply softmax to each column of scores
    P = apply_softmax(scores)
    
    ## compute the loss
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))    

    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = W.grad.numpy()
    grads['b'] = b.grad.numpy()

    return grads    


def MaxAbsoluteError(a, b):
    return np.max(np.abs(a - b))


def MaxRelativeError(a, b, eps=1e-10):
    return np.max(np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b)))

def MiniBatchGD(X, Y, y, X_val, Y_val, y_val, GDparams, init_net, lam):
    n_batch = GDparams["n_batch"]
    eta = GDparams["eta"]
    n_epochs = GDparams["n_epochs"]

    net = {
        "W": init_net["W"].copy(),
        "b": init_net["b"].copy()
    }

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_cost": [],
        "val_cost": [],
        "train_acc": [],
        "val_acc": []
    }

    n = X.shape[1]

    for epoch in range(n_epochs):
        for j in range(0, n, n_batch):
            j_end = j + n_batch

            X_batch = X[:, j:j_end]
            Y_batch = Y[:, j:j_end]

            P_batch = ApplyNetwork(X_batch, net)
            grads = BackwardPass(X_batch, Y_batch, P_batch, net, lam)

            net["W"] -= eta * grads["W"]
            net["b"] -= eta * grads["b"]

        # compute once
        P_train = ApplyNetwork(X, net)
        P_val = ApplyNetwork(X_val, net)

        train_loss = ComputeLoss(P_train, y)
        val_loss = ComputeLoss(P_val, y_val)

        train_cost = ComputeCost(P_train, y, net, lam)
        val_cost = ComputeCost(P_val, y_val, net, lam)

        train_acc = ComputeAccuracy(P_train, y)
        val_acc = ComputeAccuracy(P_val, y_val)

        # store everything
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_cost"].append(train_cost)
        history["val_cost"].append(val_cost)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)


        print(f"Epoch {epoch+1}/{n_epochs} | "
            f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | "
            f"train cost: {train_cost:.4f} | val cost: {val_cost:.4f} | "
            f"train acc: {train_acc:.4f} | val acc: {val_acc:.4f}")

    return net, history

def ComputeCost(P, y, network, lam):
    loss = ComputeLoss(P, y)
    W = network["W"]
    reg = lam * np.sum(W ** 2)
    return loss + reg

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

    L = ComputeLoss(P, trainy[:100])

    acc = ComputeAccuracy(P, trainy[:100])

    lam = 0.0
    grads = BackwardPass(trainX[:, 0:100], trainY[:, 0:100], P, net, lam)

    rng = np.random.default_rng(42)

    d_small = 10
    n_small = 3
    lam = 0.0

    small_net = {}
    small_net["W"] = 0.01 * rng.standard_normal((10, d_small))
    small_net["b"] = np.zeros((10, 1))

    X_small = trainX[0:d_small, 0:n_small]
    Y_small = trainY[:, 0:n_small]
    y_small = trainy[0:n_small]

    P_small = ApplyNetwork(X_small, small_net)

    my_grads = BackwardPass(X_small, Y_small, P_small, small_net, lam)
    torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net)

    GDparams = {
        "n_batch": 100,
        "eta": 0.001,
        "n_epochs": 40
    }

    trained_net, history = MiniBatchGD(
        trainX, trainY, trainy,
        valX, valY, valy,
        GDparams,
        net,
        lam=0.0
    )

    print("\nGradient check:")
    print("max abs error W:", MaxAbsoluteError(my_grads["W"], torch_grads["W"]))
    print("max abs error b:", MaxAbsoluteError(my_grads["b"], torch_grads["b"]))

    print("max rel error W:", MaxRelativeError(my_grads["W"], torch_grads["W"]))
    print("max rel error b:", MaxRelativeError(my_grads["b"], torch_grads["b"]))

    print("\nBackwardPass check:")
    print("grad_W shape:", grads["W"].shape)
    print("grad_b shape:", grads["b"].shape)

    print("\nAccuracy check:")
    print("Accuracy:", acc)

    print("\nLoss check:")
    print("Loss:", L)

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