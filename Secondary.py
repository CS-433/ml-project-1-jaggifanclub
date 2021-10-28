import numpy as np
from proj1_helpers import*

def compute_loss_MSE(y, tX, w):
    """Computes MSE"""
    e = y.reshape(-1,1) - tX@(w.reshape(-1, 1))
    loss_MSE = (e.T@e).item()/(2*y.size)
    return loss_MSE

def sigmoid(t):
    sig = np.empty(t.shape, dtype=np.float64) # initialization
    sig[np.logical_and(t<1000, t>-1000)] = (1.0 / (1.0 + np.exp(-t)))[np.logical_and(t<1000, t>-1000)] # 1/(1+e^(-t))
    sig[t>=1000] = 1.0 # fix numerical errors: if t>=1000, then e^(-t) < 10^434, then we assume 1/(1+e^(-t))=1
    sig[t<=-1000] = 0.0 # fix numerical errors: if t<=1000, then e^(-t) > 10^434, then we assume 1/(1+e^(-t))=0
    return sig

def compute_loss_NLL(y, tX, w, lambda_=0.0):
    y = y.reshape(-1, 1)
    tX = tX.reshape(tX.shape[0], -1)
    w = w.reshape(-1, 1)
    t = -tX@w # exponent
    log_part = np.empty(t.shape, dtype=np.float64) # initialization
    log_part[t<1000] = np.log(1 + np.exp(t))[t<1000] # log(1+e^t)
    log_part[t>=1000] = t[t>=1000] # fix numerical errors: if t>=1000, then e^t > 10^434, then we assume log(1+e^t)=t
    loss = 0.5*(1.0-y.T)@t - log_part.sum() + lambda_*(w**2).sum()
    return loss

def compute_gradient_NLL(y, tX, w, lambda_=0.0):
    y = y.reshape(-1, 1)
    tX = tX.reshape(tX.shape[0], -1)
    w = w.reshape(-1, 1)
    grad = tX.T @ (0.5*(y-1.0) + sigmoid(-tX @ w)) + 2*lambda_*w
    return grad

def compute_hessian_NLL(y, tX, w, lambda_=0.0):
    y = y.reshape(-1, 1)
    tX = tX.reshape(tX.shape[0], -1)
    w = w.reshape(-1, 1)
    S = (sigmoid(-tX @ w) * (1.0 - sigmoid(-tX @ w))).reshape(1,-1)
    H = -(tX.T*S)@tX + np.diag(np.full((w.shape[0],),2*lambda_))
    return H

def ada_grad(gradient, h, gamma_zero):
    """Step-size control for linear and logistic regression"""
    h+=np.power(gradient, 2)
    gamma=gamma_zero*(1/np.sqrt(h))
    return gamma, h

def compute_accuracy(y, tx, w):
    """Computes accuracy"""
    return np.mean(y == predict_labels(w, tx))

def compute_gradient_MSE(y, tx, w):
    """Computes gradient for (stochastic) gradient descent"""
    e = y - np.dot(tx, w)                        #dim = n
    gradient = -1/len(y)  *  np.dot(tx.T, e)     #dim = d
    return gradient

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Create mini-batches during GD/SGD."""
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def split_data(x, y, ratio, seed=1):
    """Splits the data into 2 sets."""
    # set seed
    np.random.seed(seed)
    
    #get split shuffled indexes 
    nb_row = len(y)
    idx = np.random.permutation(nb_row)
    limit = int(ratio*nb_row)
    train_idx = idx[:limit]
    test_idx = idx[limit:]
    
    # split data
    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    return x_train, y_train, x_test, y_test

def build_poly(X, degree):
    '''
    Polynomial basis functions for input data x.
    If the bias column [column of ones] is not already added in x, automatic addition of the bias column.
    :param x: data [n_samples x n_dim] or [n_samples x (n_dim+1)] if the bias is already added
    :param degree: maximum degree computed [int]
    :return: polynomial data [n_samples x (n_dim*degree+1)]
    '''
    if np.all(X[:,0]==np.ones(X.shape[0])) == False: X = np.c_[np.ones(X.shape[0]), X]
    X_poly = X[:,0].reshape(-1,1)
    for param in range(1,X.shape[1]):
        new_part = np.power(X[:,param].reshape(-1, 1), np.arange(1, degree + 1).reshape(1, -1))
        X_poly = np.hstack((X_poly, new_part))
    return X_poly
