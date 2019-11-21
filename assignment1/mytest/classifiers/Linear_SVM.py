import numpy as np


def svm_loss(W, X, y, reg):
    score = X @ W
    right_classes_score = score[range(X.shape[0]), y]
    right_classes_score = np.reshape(right_classes_score, (right_classes_score.shape[0], 1))
    margin = np.maximum(0, score - right_classes_score + 1)
    # 把正确类别对应位置的margin变为0，相当于除去y=j的项
    margin[range(X.shape[0]), y] = 0
    first = np.sum(margin) / X.shape[0]
    second = reg * np.sum(W ** 2)
    loss = first + second

    #此时margin的j列和X的i列点积得到的W（ij）的值。而margin的j列就是X的i列被叠加的次数。
    #下面对应https://zhuanlan.zhihu.com/p/45753542中公式一的求导
    margin[margin > 0] = 1
    #对应公式二的求导
    row_sum = np.sum(margin, axis=1)
    margin[range(X.shape[0]), y] = -row_sum
    #求出第一部分求和
    grad = X.T @ margin

    grad = grad / X.shape[0] + 2 * reg * W
    return loss, grad