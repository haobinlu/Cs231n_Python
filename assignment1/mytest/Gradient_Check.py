import random
import numpy as np

def GradientCheck(f, W, grad, num_check = 10, h = 1e-5):
    for i in range(num_check):
        index = tuple([random.randrange(m) for m in W.shape])
        odval = W[index]
        W[index] = odval + h
        losspuls = f(W)
        W[index] = odval - h
        lossdec = f(W)
        W[index] = odval
        numgrad = (losspuls - lossdec) / (2 * h)
        norm = np.linalg.norm(grad[index] - numgrad)
        print('compute_grad=%f analyse_grad=%f norm=%e' % (numgrad, grad[index], norm))

def gradient_check_nn(f, W, verbose=True, h=0.00001):
    num_grad = np.zeros_like(W)
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        temp = W[ix]
        W[ix] = temp + h
        plus = f(W)
        W[ix] = temp - h
        dec = f(W)
        W[ix] = temp

        num_grad[ix] = (plus - dec) / (2 * h)

        if verbose == 'Ture':
            print(ix, num_grad[ix])
        it.iternext()

    return num_grad


