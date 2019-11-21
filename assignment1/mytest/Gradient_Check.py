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
