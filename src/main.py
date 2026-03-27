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

def ComputeGradsWithTorch(X, y, network_params, lam=0.0):

    Xt = torch.from_numpy(X)

    W = torch.tensor(network_params['W'], requires_grad=True)
    b = torch.tensor(network_params['b'], requires_grad=True)

    N = X.shape[1]

    scores = torch.matmul(W, Xt) + b
    apply_softmax = torch.nn.Softmax(dim=0)
    P = apply_softmax(scores)
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))
    cost = loss + lam * torch.sum(W * W)

    cost.backward()

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

def PlotHistory(history, title="", save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # Cost
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_cost"], label="train cost")
    plt.plot(epochs, history["val_cost"], label="val cost")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Cost")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.close()


def VisualizeWeights(network, save_path=None):
    W = network["W"]  # (10, 3072)
    Ws = W.T.reshape((32, 32, 3, 10), order="F")
    W_im = np.transpose(Ws, (1, 0, 2, 3))

    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    _, axs = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axs.flat):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - w_im.min()) / (w_im.max() - w_im.min())
        ax.imshow(w_im_norm)
        ax.set_title(class_names[i], fontsize=9)
        ax.axis("off")

    plt.suptitle("Learnt weight templates")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved weights: {save_path}")
    plt.close()

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    data_dir = ROOT / "Datasets" / "cifar-10-python" / "cifar-10-batches-py"
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    trainX, trainY, trainy = LoadBatch(data_dir / "data_batch_1")
    valX,   valY,   valy   = LoadBatch(data_dir / "data_batch_2")
    testX,  testY,  testy  = LoadBatch(data_dir / "test_batch")

    mean_X = np.mean(trainX, axis=1, keepdims=True)
    std_X  = np.std(trainX,  axis=1, keepdims=True)

    trainX = NormalizeData(trainX, mean_X, std_X)
    valX   = NormalizeData(valX,   mean_X, std_X)
    testX  = NormalizeData(testX,  mean_X, std_X)

    K = 10
    d = trainX.shape[0]

    rng = np.random.default_rng(42)
    d_small, n_small = 10, 3

    small_net = {
        "W": 0.01 * rng.standard_normal((K, d_small)),
        "b": np.zeros((K, 1))
    }

    X_small = trainX[0:d_small, 0:n_small]
    Y_small = trainY[:, 0:n_small]
    y_small = trainy[0:n_small]

    P_small   = ApplyNetwork(X_small, small_net)
    my_grads  = BackwardPass(X_small, Y_small, P_small, small_net, lam=0.0)
    torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net)

    print("-- Gradient check (lam=0) -------------------------")
    print(f"  max abs error  W: {MaxAbsoluteError(my_grads['W'], torch_grads['W']):.2e}")
    print(f"  max abs error  b: {MaxAbsoluteError(my_grads['b'], torch_grads['b']):.2e}")
    print(f"  max rel error  W: {MaxRelativeError(my_grads['W'], torch_grads['W']):.2e}")
    print(f"  max rel error  b: {MaxRelativeError(my_grads['b'], torch_grads['b']):.2e}")

    # repeat with lam > 0 to check regularisation term
    lam_check = 0.1
    my_grads_reg    = BackwardPass(X_small, Y_small, P_small, small_net, lam=lam_check)
    torch_grads_reg = ComputeGradsWithTorch(X_small, y_small, small_net, lam=lam_check)

    print(f"\n-- Gradient check (lam={lam_check}) -------------------------")
    print(f"  max abs error  W: {MaxAbsoluteError(my_grads_reg['W'], torch_grads_reg['W']):.2e}")
    print(f"  max abs error  b: {MaxAbsoluteError(my_grads_reg['b'], torch_grads_reg['b']):.2e}")
    print(f"  max rel error  W: {MaxRelativeError(my_grads_reg['W'], torch_grads_reg['W']):.2e}")
    print(f"  max rel error  b: {MaxRelativeError(my_grads_reg['b'], torch_grads_reg['b']):.2e}")

    # Training for all 4 given configurations
    configs = [
        {"lam": 0.0, "eta": 0.1,   "n_epochs": 40, "n_batch": 100},
        {"lam": 0.0, "eta": 0.001, "n_epochs": 40, "n_batch": 100},
        {"lam": 0.1, "eta": 0.001, "n_epochs": 40, "n_batch": 100},
        {"lam": 1.0, "eta": 0.001, "n_epochs": 40, "n_batch": 100},
    ]

    print("\n-- Training---------------------------------")
    for cfg in configs:
        lam      = cfg["lam"]
        eta      = cfg["eta"]
        n_epochs = cfg["n_epochs"]
        n_batch  = cfg["n_batch"]

        label = f"lam{lam}_eta{eta}_epochs{n_epochs}_batch{n_batch}"
        print(f"\n  Config: {label}")

        init_net = InitNetwork(K, d, seed=42)
        GDparams = {"n_batch": n_batch, "eta": eta, "n_epochs": n_epochs}

        trained_net, history = MiniBatchGD(trainX, trainY, trainy, valX,   valY,   valy, GDparams, init_net, lam)

        # test accuracy
        P_test   = ApplyNetwork(testX, trained_net)
        test_acc = ComputeAccuracy(P_test, testy)
        print(f"  Final test accuracy: {test_acc*100:.2f}%")

        PlotHistory(history, title=f"lam={lam}, eta={eta}", save_path=figures_dir / f"history_{label}.png")

        VisualizeWeights(trained_net,save_path=figures_dir / f"weights_{label}.png")