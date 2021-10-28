import numpy as np
from proj1_helpers import*

def compute_loss_MSE(y, tX, w):
    """Computes MSE"""
    e = y.reshape(-1,1) - tX@(w.reshape(-1, 1))
    loss_MSE = (e.T@e).item()/(2*y.size)
    return loss_MSE

def compute_loss_NLL(y, tx, w):
    """Compute the loss: negative log likelihood.    -   A CHANGER, PAS LA BONNE LOSS POUR (-1,1)"""
    sig = sigmoid(tx.dot(w))
    loss = - np.sum( y*np.log(sig)  +  (1 - y) * np.log(1-sig) )
    return loss

def ada_grad(gradient, h, gamma_zero):
    """Step-size control for linear and logistic regression"""
    h+=np.power(gradient, 2)
    gamma=gamma_zero*(1/np.sqrt(h))
    return gamma, h

def compute_accuracy(y, tx, w):
    """Computes accuracy"""
    return np.mean(y == predict_labels(w, tx))

def compute_gradient(y, tx, w):
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
