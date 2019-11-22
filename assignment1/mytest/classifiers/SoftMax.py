import numpy as np



def softmax_loss_vectorized(W, X, y, reg):
    #计算normalize过后的pre_Y
    pre_Y = X @ W
    pre_Y_max = np.max(pre_Y, axis=1)
    pre_Y_max = np.reshape(pre_Y_max, (pre_Y_max.shape[0], 1))
    pre_Y -= pre_Y_max

    #计算grad
    P = np.exp(pre_Y) / np.sum(np.exp(pre_Y), axis=1)[..., np.newaxis]
    print(P)
    P[range(y.shape[0]), y] = P[range(y.shape[0]), y] - 1
    dw1 = X.T @ P
    dw = dw1 / X.shape[0] + 2 * reg * W

    #计算损失值
    first = -pre_Y[range(y.shape[0]), y]
    second = np.log(np.sum(np.exp(pre_Y), axis= 1))
    loss = np.sum((first + second)) / X.shape[0] + reg * np.sum(W ** 2)
    return loss, dw