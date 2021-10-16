import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import*
"""REQUIRED FUNCTIONS"""

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm using MSE loss."""
    # Define initial loss and weights
    loss = compute_loss_MSE(y, tx, w)
    w = initial_w
    for n_iter in range(max_iters):                                    #''' TODO: Learning rate update ?'''
        loss = compute_loss_MSE(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
#        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size = 1):
    """Stochastic gradient descent algorithm using MSE loss."""
    # Define parameters to store w and loss
    loss = compute_loss_MSE(y, tx, w)
    w = initial_w
    for n_iter in range(max_iters):
        generator = batch_iter(y, tx, batch_size)                      #''' TODO: A OPTIMISER ?'''
        y_sub, tx_sub = next(generator)
        loss = compute_loss_MSE(y_sub, tx_sub, w)
        stoch_gradient = compute_gradient(y_sub, tx_sub, w)
        w = w - gamma * stoch_gradient
#        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    loss = compute_loss_MSE(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss = compute_loss_MSE(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    a = tx.T.dot(tx) + 2*len(y)*lambda_*np.eye(len(tx.T))
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss = compute_loss_MSE(y, tx, w)
    return w, loss





"""ADITIONAL FUNCTIONS"""

"""Functions used to compute the loss or accuracy."""

def compute_loss_MSE(y, tx, w):
    loss = 1/(2*len(y)) * np.sum((y - np.dot(tx, w))**2)
    return loss

def compute_loss_MAE(y, tx, w):
    loss = 1/(len(y)) * np.sum(np.abs(y - np.dot(tx, w)))
    return loss

def accuracy(y, weights, data):
    return np.mean(y == predict_labels(weights, data))

"""Functions used to compute the gradient for GD/SGD."""

def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)                        #dim = n
    gradient = -1/len(y)  *  np.dot(tx.T, e)     #dim = d
    return gradient


"""Functions used to create mini-batches during GD/SGD."""

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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


"""Functions used to split the data into 2 sets."""

def split_data(x, y, ratio, seed=1):
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


"""Functions used to build a polynomial feature of x as matrix"""

def build_poly(x, degree):
    ''' Polynomial basis functions for input data x, for j=0 up to j=degree. '''
    x_poly = x[:,0].reshape(-1,1)
    for param in range(1,x.shape[1]):
        new_part = np.power(x[:,param].reshape(-1, 1), np.arange(1, degree + 1).reshape(1, -1))
        x_poly = np.hstack((x_poly, new_part))
    return x_poly

"""Functions used to get training/test loss on the kth fold, for a feature x matrix of degree x, for a given model"""

def cross_validation(y, x, k_indices, k, lambda_, degree, model):
    #separate line of index taken for test split
    train_folds = list(range(k_indices.shape[0]))
    train_folds.remove(k)
    train_idx = np.concatenate(([k_indices[fold,:] for fold in train_folds]))
    test_idx = k_indices[k,:]
    # ***************************************************
    feat_matrix_tr = build_poly(x[train_idx], degree)
    feat_matrix_te = build_poly(x[test_idx], degree)
    y_tr = y[train_idx]
    y_te = y[test_idx]
    # ***************************************************
    loss_tr, weights = model(y_tr, feat_matrix_tr, lambda_)
    loss_te = compute_mse(y_te, feat_matrix_te, weights)
    return loss_tr, loss_te


"""Functions used to plot losses and parameters for model selection"""

def plot_lambda_vs_loss(lambdas, err_tr, err_te, err_type = 'mse'):
    """visualization the curves of mse/accuracy."""
    best_idx = np.argmin(err_te)
    
    plt.semilogx(lambdas, err_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambdas, err_te, marker=".", color='r', label='test error')
    plt.axvline(lambdas[best_idx], color = 'k', ls = '--', alpha = 0.5, label = 'best lambda')
    plt.xlabel("lambda")
    plt.ylabel(err_type)
    plt.title("Best lambda selection")
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()
    
def plot_degree_vs_loss(degrees, err_tr, err_te, err_type = 'mse'):
    """visualization the curves of mse/accuracy."""
    best_idx = np.argmin(err_te)
    
    plt.plot(degrees, err_tr, marker=".", color='b', label='train error')
    plt.plot(degrees, err_te, marker=".", color='r', label='test error')
    plt.axvline(degrees[best_idx], color = 'k', ls = '--', alpha = 0.5, label = 'best degree')
    plt.xlabel("degree")
    plt.ylabel(err_type)
    plt.title("Best degree selection")
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()
    
def plot_boxplot(losses, model_names, err_type = 'mse'):
    """visualisation of the variance of models"""
    plt.boxplot(losses, labels = model_names)
    plt.title('Boxplot of models (' + str(np.array(losses).shape[1]) + ' folds)')
    plt.ylabel(err_type)
    plt.show()