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


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def ApplyNetworkSigmoid(X, network):
    W = network["W"]
    b = network["b"]
    return sigmoid(W @ X + b)


def ComputeLossBCE(P, y):
    """Mean multiple-binary-cross-entropy loss. y is integer labels (1-D)."""
    K, n = P.shape
    Y = np.zeros_like(P)
    Y[y, np.arange(n)] = 1.0
    P_c = np.clip(P, 1e-15, 1 - 1e-15)
    L = -np.mean(np.sum((1 - Y) * np.log(1 - P_c) + Y * np.log(P_c), axis=0)) / K
    return L


def BackwardPassBCE(X, Y, P, network, lam):
    """Gradient of mean BCE loss + L2.  ∂l/∂s = (1/K)(p − y)."""
    W = network["W"]
    n = X.shape[1]
    K = Y.shape[0]

    G = (P - Y) / K                               # (K, n)
    grad_W = (G @ X.T) / n + 2 * lam * W
    grad_b = np.sum(G, axis=1, keepdims=True) / n

    return {"W": grad_W, "b": grad_b}


def ComputeGradsWithTorchBCE(X, y, network_params, lam=0.0):
    Xt = torch.from_numpy(X)
    W  = torch.tensor(network_params['W'], requires_grad=True)
    b  = torch.tensor(network_params['b'], requires_grad=True)
    K, n = W.shape[0], X.shape[1]

    Y = np.zeros((K, n))
    Y[y, np.arange(n)] = 1.0
    Yt = torch.from_numpy(Y)

    scores = torch.matmul(W, Xt) + b
    P = torch.sigmoid(scores)
    P_c = torch.clamp(P, 1e-15, 1 - 1e-15)
    bce = -torch.mean(
        torch.sum((1 - Yt) * torch.log(1 - P_c) + Yt * torch.log(P_c), dim=0)
    ) / K
    cost = bce + lam * torch.sum(W * W)
    cost.backward()

    return {"W": W.grad.numpy(), "b": b.grad.numpy()}


def MiniBatchGD(X, Y, y, X_val, Y_val, y_val, GDparams, init_net, lam,
                inds_flip=None, rng=None,
                apply_fn=None, loss_fn=None, backward_fn=None):
    if apply_fn    is None: apply_fn    = ApplyNetwork
    if loss_fn     is None: loss_fn     = ComputeLoss
    if backward_fn is None: backward_fn = BackwardPass

    n_batch      = GDparams["n_batch"]
    eta          = GDparams["eta"]
    n_epochs     = GDparams["n_epochs"]
    decay_factor = GDparams.get("decay_factor", 1.0)
    decay_every  = GDparams.get("decay_every",  0)

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

            X_batch = X[:, j:j_end].copy()
            Y_batch = Y[:, j:j_end]

            if inds_flip is not None and rng is not None:
                flip_mask = rng.random(X_batch.shape[1]) < 0.5
                X_flipped = X_batch[inds_flip, :]
                X_batch[:, flip_mask] = X_flipped[:, flip_mask]

            P_batch = apply_fn(X_batch, net)
            grads = backward_fn(X_batch, Y_batch, P_batch, net, lam)

            net["W"] -= eta * grads["W"]
            net["b"] -= eta * grads["b"]

        # compute once
        P_train = apply_fn(X, net)
        P_val   = apply_fn(X_val, net)

        train_loss = loss_fn(P_train, y)
        val_loss   = loss_fn(P_val,   y_val)

        train_cost = train_loss + lam * np.sum(net["W"] ** 2)
        val_cost   = val_loss   + lam * np.sum(net["W"] ** 2)

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
            f"train acc: {train_acc:.4f} | val acc: {val_acc:.4f} | eta: {eta:.6f}")

        if decay_every > 0 and (epoch + 1) % decay_every == 0:
            eta *= decay_factor
            print(f"  [LR decay] eta → {eta:.6f}")

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


def PlotConfidenceHistogram(p_correct, p_wrong, title="", save_path=None):
    """Histograms of P(true class) for correctly vs incorrectly classified examples."""
    _, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axs[0].hist(p_correct, bins=20, range=(0, 1), color="steelblue")
    axs[0].set_title(f"Correct ({len(p_correct)})")
    axs[0].set_xlabel("P(true class)")
    axs[1].hist(p_wrong, bins=20, range=(0, 1), color="tomato")
    axs[1].set_title(f"Incorrect ({len(p_wrong)})")
    axs[1].set_xlabel("P(true class)")
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved histogram: {save_path}")
    plt.close()


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    data_dir = ROOT / "Datasets" / "cifar-10-python" / "cifar-10-batches-py"
    figures_dir       = ROOT / "figures"
    figures_bonus_dir = ROOT / "figures" / "bonus"
    figures_dir.mkdir(exist_ok=True)
    figures_bonus_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # PHASE 1 — Required configs                                          #
    # Train: data_batch_1 (10k), Val: data_batch_2, no flip augmentation #
    # ------------------------------------------------------------------ #
    p1_trainX, p1_trainY, p1_trainy = LoadBatch(data_dir / "data_batch_1")
    p1_valX,   p1_valY,   p1_valy   = LoadBatch(data_dir / "data_batch_2")
    testX,     testY,     testy     = LoadBatch(data_dir / "test_batch")

    p1_mean_X = np.mean(p1_trainX, axis=1, keepdims=True)
    p1_std_X  = np.std(p1_trainX,  axis=1, keepdims=True)

    p1_trainX = NormalizeData(p1_trainX, p1_mean_X, p1_std_X)
    p1_valX   = NormalizeData(p1_valX,   p1_mean_X, p1_std_X)
    p1_testX  = NormalizeData(testX,     p1_mean_X, p1_std_X)

    K = 10
    d = p1_trainX.shape[0]

    # Gradient checks use phase-1 data
    rng = np.random.default_rng(42)
    d_small, n_small = 10, 3

    small_net = {
        "W": 0.01 * rng.standard_normal((K, d_small)),
        "b": np.zeros((K, 1))
    }

    X_small = p1_trainX[0:d_small, 0:n_small]
    Y_small = p1_trainY[:, 0:n_small]
    y_small = p1_trainy[0:n_small]

    P_small     = ApplyNetwork(X_small, small_net)
    my_grads    = BackwardPass(X_small, Y_small, P_small, small_net, lam=0.0)
    torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net)

    print("-- Gradient check (lam=0) -------------------------")
    print(f"  max abs error  W: {MaxAbsoluteError(my_grads['W'], torch_grads['W']):.2e}")
    print(f"  max abs error  b: {MaxAbsoluteError(my_grads['b'], torch_grads['b']):.2e}")
    print(f"  max rel error  W: {MaxRelativeError(my_grads['W'], torch_grads['W']):.2e}")
    print(f"  max rel error  b: {MaxRelativeError(my_grads['b'], torch_grads['b']):.2e}")

    lam_check = 0.1
    my_grads_reg    = BackwardPass(X_small, Y_small, P_small, small_net, lam=lam_check)
    torch_grads_reg = ComputeGradsWithTorch(X_small, y_small, small_net, lam=lam_check)

    print(f"\n-- Gradient check (lam={lam_check}) -------------------------")
    print(f"  max abs error  W: {MaxAbsoluteError(my_grads_reg['W'], torch_grads_reg['W']):.2e}")
    print(f"  max abs error  b: {MaxAbsoluteError(my_grads_reg['b'], torch_grads_reg['b']):.2e}")
    print(f"  max rel error  W: {MaxRelativeError(my_grads_reg['W'], torch_grads_reg['W']):.2e}")
    print(f"  max rel error  b: {MaxRelativeError(my_grads_reg['b'], torch_grads_reg['b']):.2e}")

    configs = [
        {"lam": 0.0, "eta": 0.1,   "n_epochs": 40, "n_batch": 100},
        {"lam": 0.0, "eta": 0.001, "n_epochs": 40, "n_batch": 100},
        {"lam": 0.1, "eta": 0.001, "n_epochs": 40, "n_batch": 100},
        {"lam": 1.0, "eta": 0.001, "n_epochs": 40, "n_batch": 100},
    ]

    print("\n-- Phase 1: required configs (batch_1 train, batch_2 val, no augmentation) --")
    for cfg in configs:
        lam      = cfg["lam"]
        eta      = cfg["eta"]
        n_epochs = cfg["n_epochs"]
        n_batch  = cfg["n_batch"]

        label = f"lam{lam}_eta{eta}_epochs{n_epochs}_batch{n_batch}"
        print(f"\n  Config: {label}")

        init_net = InitNetwork(K, d, seed=42)
        GDparams = {"n_batch": n_batch, "eta": eta, "n_epochs": n_epochs}

        trained_net, history = MiniBatchGD(
            p1_trainX, p1_trainY, p1_trainy,
            p1_valX,   p1_valY,   p1_valy,
            GDparams, init_net, lam,
            inds_flip=None, rng=None
        )

        P_test   = ApplyNetwork(p1_testX, trained_net)
        test_acc = ComputeAccuracy(P_test, testy)
        print(f"  Final test accuracy: {test_acc*100:.2f}%")

        PlotHistory(history, title=f"lam={lam}, eta={eta}",
                    save_path=figures_dir / f"history_{label}.png")
        VisualizeWeights(trained_net,
                         save_path=figures_dir / f"weights_{label}.png")

    # ------------------------------------------------------------------ #
    # PHASE 2 — Bonus                                                     #
    # Train: all 5 batches minus last 1k, Val: last 1k, flip augmentation #
    # ------------------------------------------------------------------ #
    all_X, all_Y, all_y = [], [], []
    for i in range(1, 6):
        X_i, Y_i, y_i = LoadBatch(data_dir / f"data_batch_{i}")
        all_X.append(X_i); all_Y.append(Y_i); all_y.append(y_i)
    all_X = np.concatenate(all_X, axis=1)
    all_Y = np.concatenate(all_Y, axis=1)
    all_y = np.concatenate(all_y, axis=0)

    trainX, trainY, trainy = all_X[:, :-1000], all_Y[:, :-1000], all_y[:-1000]
    valX,   valY,   valy   = all_X[:, -1000:], all_Y[:, -1000:], all_y[-1000:]

    mean_X = np.mean(trainX, axis=1, keepdims=True)
    std_X  = np.std(trainX,  axis=1, keepdims=True)

    trainX = NormalizeData(trainX, mean_X, std_X)
    valX   = NormalizeData(valX,   mean_X, std_X)
    bonusTestX = NormalizeData(testX, mean_X, std_X)

    aa = np.int32(np.arange(32)).reshape((32, 1))
    bb = np.int32(np.arange(31, -1, -1)).reshape((32, 1))
    vv = np.tile(32 * aa, (1, 32))
    ind_flip = vv.reshape((32 * 32, 1)) + np.tile(bb, (32, 1))
    inds_flip = np.vstack((ind_flip, 1024 + ind_flip))
    inds_flip = np.vstack((inds_flip, 2048 + ind_flip)).squeeze()  # (3072,)

    print("\n-- Phase 2: grid search ------------------------------------")
    grid_etas    = [0.01, 0.005, 0.001]
    grid_lams    = [0.01, 0.05,  0.1]
    grid_batches = [50,   100,   200]

    best_val_acc = -1.0
    best_cfg_gs  = None

    for gs_eta in grid_etas:
        for gs_lam in grid_lams:
            for gs_batch in grid_batches:
                gs_label = f"gs_eta{gs_eta}_lam{gs_lam}_batch{gs_batch}"
                print(f"\n  {gs_label}")

                gs_params = {
                    "n_batch":      gs_batch,
                    "eta":          gs_eta,
                    "n_epochs":     40,
                    "decay_factor": 0.1,
                    "decay_every":  20,
                }
                gs_net = InitNetwork(K, d, seed=42)
                gs_trained, gs_history = MiniBatchGD(
                    trainX, trainY, trainy, valX, valY, valy,
                    gs_params, gs_net, gs_lam,
                    inds_flip=inds_flip, rng=np.random.default_rng(42)
                )
                val_acc_final = gs_history["val_acc"][-1]
                print(f"  val acc: {val_acc_final*100:.2f}%")

                if val_acc_final > best_val_acc:
                    best_val_acc = val_acc_final
                    best_cfg_gs  = gs_label

    print(f"\n-- Grid search best: {best_cfg_gs}  val acc: {best_val_acc*100:.2f}%")

    print("\n-- Best grid config test evaluation ------------------")
    gs_best_params = {"n_batch": 200, "eta": 0.005, "n_epochs": 40,
                      "decay_factor": 0.1, "decay_every": 20}
    gs_best_net = InitNetwork(K, d, seed=42)
    trained_best, _ = MiniBatchGD(
        trainX, trainY, trainy, valX, valY, valy,
        gs_best_params, gs_best_net, lam=0.01,
        inds_flip=inds_flip, rng=np.random.default_rng(42)
    )
    P_test_best = ApplyNetwork(bonusTestX, trained_best)
    print(f"  Best grid config test acc: {ComputeAccuracy(P_test_best, testy)*100:.2f}%")

    # --- BCE gradient check ---
    P_bce_small  = ApplyNetworkSigmoid(X_small, small_net)
    Y_small_oh   = np.zeros((K, n_small))
    Y_small_oh[y_small, np.arange(n_small)] = 1.0
    my_grads_bce    = BackwardPassBCE(X_small, Y_small_oh, P_bce_small, small_net, lam=0.0)
    torch_grads_bce = ComputeGradsWithTorchBCE(X_small, y_small, small_net, lam=0.0)

    print("\n-- BCE gradient check (lam=0) -------------------------")
    print(f"  max abs error  W: {MaxAbsoluteError(my_grads_bce['W'], torch_grads_bce['W']):.2e}")
    print(f"  max abs error  b: {MaxAbsoluteError(my_grads_bce['b'], torch_grads_bce['b']):.2e}")
    print(f"  max rel error  W: {MaxRelativeError(my_grads_bce['W'], torch_grads_bce['W']):.2e}")
    print(f"  max rel error  b: {MaxRelativeError(my_grads_bce['b'], torch_grads_bce['b']):.2e}")

    print("\n-- BCE training -----------------------------------")
    bce_params = {"n_batch": 50, "eta": 0.01, "n_epochs": 40,
                  "decay_factor": 0.1, "decay_every": 20}
    bce_net = InitNetwork(K, d, seed=42)
    trained_bce, history_bce = MiniBatchGD(
        trainX, trainY, trainy, valX, valY, valy,
        bce_params, bce_net, lam=0.01,
        inds_flip=inds_flip, rng=np.random.default_rng(42),
        apply_fn=ApplyNetworkSigmoid,
        loss_fn=ComputeLossBCE,
        backward_fn=BackwardPassBCE
    )

    print("\n-- Softmax comparison run -------------------------")
    sm_params = {"n_batch": 50, "eta": 0.001, "n_epochs": 40,
                 "decay_factor": 0.1, "decay_every": 20}
    sm_net = InitNetwork(K, d, seed=42)
    trained_sm, history_sm = MiniBatchGD(
        trainX, trainY, trainy, valX, valY, valy,
        sm_params, sm_net, lam=0.01,
        inds_flip=inds_flip, rng=np.random.default_rng(42)
    )

    P_test_bce = ApplyNetworkSigmoid(bonusTestX, trained_bce)
    P_test_sm  = ApplyNetwork(bonusTestX,        trained_sm)
    acc_bce = ComputeAccuracy(P_test_bce, testy)
    acc_sm  = ComputeAccuracy(P_test_sm,  testy)
    print(f"\n  BCE  test accuracy: {acc_bce*100:.2f}%")
    print(f"  Softmax test accuracy: {acc_sm*100:.2f}%")

    PlotHistory(history_bce, title="BCE (sigmoid), eta=0.01, lam=0.01",
                save_path=figures_bonus_dir / "history_bce.png")
    PlotHistory(history_sm,  title="Softmax CE,    eta=0.001, lam=0.01",
                save_path=figures_bonus_dir / "history_softmax_cmp.png")

    def true_class_probs(P, y):
        return P[y, np.arange(P.shape[1])]

    p_true_bce = true_class_probs(P_test_bce, testy)
    correct_bce = np.argmax(P_test_bce, axis=0) == testy
    PlotConfidenceHistogram(p_true_bce[correct_bce], p_true_bce[~correct_bce],
                            title="Sigmoid BCE — P(true class)",
                            save_path=figures_bonus_dir / "histogram_bce.png")

    p_true_sm = true_class_probs(P_test_sm, testy)
    correct_sm = np.argmax(P_test_sm, axis=0) == testy
    PlotConfidenceHistogram(p_true_sm[correct_sm], p_true_sm[~correct_sm],
                            title="Softmax CE — P(true class)",
                            save_path=figures_bonus_dir / "histogram_softmax.png")