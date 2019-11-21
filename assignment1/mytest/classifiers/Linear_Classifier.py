import numpy as np
from mytest.classifiers import Linear_SVM

class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, Y, reg=1e-5, iteration=100, learning_rate=1e-3, batc_size=200, verbose=False):
        if self.W is None:
            num_classes = np.max(Y) + 1
            self.W = np.random.randn(X.shape[1], num_classes) * 0.001

        loss_history = []
        for it in range(iteration):
            batch_index = np.random.choice(range(X.shape[0]), batc_size)
            X_batch = X[batch_index]
            Y_batch = Y[batch_index]
            loss, grad = self.loss(X_batch, Y_batch, reg)
            loss_history.append(loss)
            self.W += -(learning_rate * grad)
            if verbose == True and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, iteration, loss))
        return loss_history

    def predict(self, X):
        score = X @ self.W
        predict = np.argmax(score, axis=1)
        return predict

class LinearSVM(LinearClassifier):
    #这是LinearClassifier的子类
    def loss(self, X_batch, Y_batch, reg):
        return Linear_SVM.svm_loss(self.W, X_batch, Y_batch, reg)