import numpy as np


class KNearestNeighbor(object):

    def __init__(self):
        pass

    def train(self, X, Y):
        self.X_train = X
        self.Y_trian = Y

    def compute_distances_two_loops(self, X_test):
        num_train = self.X_train.shape[0]
        num_test = X_test.shape[0]
        dis = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dis[i, j] = np.linalg.norm(X_test[i, :] - self.X_train[j, :])
                # dis[i, j] = np.sqrt(np.sum((X_test[i, :] - self.X_train[j, :]) ** 2))
        return dis

    def compute_distances_one_loop(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.linalg.norm(X_test[i, :] - self.X_train, axis= 1)
            #dists[i, :] = np.sqrt(np.sum((X_test[i, :] - self.X_train) ** 2, axis=1))
        return dists

#这是最快的计算方式
    def compute_distances_no_loops(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists += np.sum(self.X_train ** 2, axis= 1).reshape(1, num_train)
        dists += np.sum(X_test ** 2, axis= 1).reshape(num_test, 1)
        dists += -2 * (X_test @ self.X_train.T)
        return np.sqrt(dists)


    def predict(self, dists, k):
        num = dists.shape[0]
        pre_Y = np.zeros(num)
        for i in range(num):
            Y_cloest = []
            #选出最小的K个数
            idx = np.argsort(dists[i])[0:k]
            Y_cloest = self.Y_trian[idx].astype('int64')
            #投票过程：选出票数最多的类别
            pre_Y[i] = np.argmax(np.bincount(Y_cloest))
        return pre_Y