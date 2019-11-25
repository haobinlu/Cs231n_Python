import numpy as np


class twolayer_net(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.param = {}
        self.param['W1'] = std * np.random.randn(input_size, hidden_size)
        self.param['b1'] = np.zeros(hidden_size)
        self.param['W2'] = std * np.random.randn(hidden_size, output_size)
        self.param['b2'] = np.zeros(output_size)

    def loss(self, X, Y=None, reg=1.0):
        W1, b1 = self.param['W1'], self.param['b1']
        W2, b2 = self.param['W2'], self.param['b2']

        #计算loss
        #计算过程  f1 = (W1X + b1) ---> h = relu(f1) ---> f2 = (W2h + b2) ---> P = SoftMax(f2) ---> Loss
        f1 = X @ W1 + b1
        h = np.maximum(0, f1)
        f2 = h @ W2 + b2

        #score就是f2
        score = f2
        if Y is None:
            return score
        P = np.exp(f2) / np.sum(np.exp(f2), axis=1)[..., np.newaxis]
        correct_logprobs = -np.log(P[range(P.shape[0]), Y])
        data_loss = np.sum(correct_logprobs) / X.shape[0]
        reg_loss = reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(b1 * b1) + np.sum(b2 * b2))
        L = data_loss + reg_loss

        #计算grad
        grad = {}
        #dL_W2 = dL_f2 * df2_W2(the value equals to h)
        P[range(P.shape[0]), Y] -= 1
        dL_f2 = P / X.shape[0]
        dL_W2 = h.T @ dL_f2
        grad['W2'] = dL_W2 + 2 * reg * W2

        #dL_b2 = dL_f2 * df2_b2(the value equals to 1)
        dL_b2 = dL_f2
        grad['b2'] = np.sum(dL_b2, axis=0) + 2 * reg * b2

        #dL_W1 = dL_f2 * df2_h(the value equals to W2) * dh_f1 * df1_W1(the value equals to X)
        dL_h = dL_f2 @ W2.T
        dh_f1 = h > 0
        dL_W1 = X.T @ (dh_f1 * dL_h) + 2 * reg * W1
        grad['W1'] = dL_W1

        # dL_W1 = dL_f2 * df2_h(the value equals to W2) * dh_f1 * df1_b1(the value equals to 1)
        dL_b1 = np.sum(dh_f1 * dL_h, axis=0) + 2 * reg * b1
        grad['b1'] = dL_b1

        return L, grad

    def train(self, X_train, Y_train, X_val, Y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False
              ):
        num_train = X_train.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        for it in range(num_iters):
            batch_index = np.random.choice(range(num_train), batch_size)
            X_batch = X_train[batch_index]
            Y_batch = Y_train[batch_index]
            loss, grad = self.loss(X_batch, Y_batch, reg= reg)
            loss_history.append(loss)
            for W in self.param:
                self.param[W] -= learning_rate * grad[W]

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                y_pre_train = self.predict(X_train)
                acc_train = np.mean(y_pre_train == Y_train)
                train_acc_history.append(acc_train)

                y_pre_val = self.predict(X_val)
                acc_val = np.mean(y_pre_val == Y_val)
                val_acc_history.append(acc_val)

                learning_rate *= learning_rate_decay

        return {'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,}

    def predict(self, X):
        f1 = X @ self.param['W1'] + self.param['b1']
        h = np.maximum(0, f1)
        f2 = h @ self.param['W2'] + self.param['b2']
        P = np.exp(f2) / np.sum(np.exp(f2), axis=1)[..., np.newaxis]
        y_pred = np.argmax(P, axis=1)
        return y_pred