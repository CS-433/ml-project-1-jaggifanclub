import math
import numpy as np
from Secondary import *
import matplotlib.pyplot as plt

"""""""""""""""""""""""
" REQUIRED FUNCTIONS  "
"""""""""""""""""""""""

def least_squares_GD(y, tx, initial_w, max_iters, gamma, plot=False, ada_grad=False):
    """Gradient descent algorithm using MSE loss."""
    # Define initial loss and weights
    w = initial_w.reshape(-1, 1)
    h = np.zeros(w.shape)
    loss = compute_loss_MSE(y, tx, w)
    losses=[]
    for n_iter in range(max_iters):                                    
        loss = compute_loss_MSE(y, tx, w)
        gradient = compute_gradient_MSE(y, tx, w)
        if ada_grad:
            gamma_actual, h=ada_grad(gradient, h, gamma)
            w = w - gamma_actual * gradient
        else:
            w = w - gamma * gradient
        # if n_iter%20==0:
        #     print("Gradient Descent({bi}/{ti}): loss ={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=np.round(loss,4), w0=np.round(w[0],4), w1=np.round(w[1],4)))
        if plot:
            losses.append(loss)
    if plot:
        plt.plot(losses, '-', label="AdaGrad method")
        plt.xlabel("Number of iterations")
        plt.ylabel("Average Test loss over the k folds for the best degree")
        plt.legend
        plt.show
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size = 1, plot=False, ada_grad=False):
    """Stochastic gradient descent algorithm using MSE loss."""
    # Define parameters to store w and loss
    w = initial_w.reshape(-1, 1)
    h = np.zeros(w.shape)
    loss = compute_loss_MSE(y, tx, w)
    losses=[]
    for n_iter in range(max_iters):
        generator = batch_iter(y, tx, batch_size)
        y_sub, tx_sub = next(generator)
        loss = compute_loss_MSE(y_sub, tx_sub, w)
        stoch_gradient = compute_gradient_MSE(y_sub, tx_sub, w)
        if ada_grad:
            gamma_actual, h=ada_grad(stoch_gradient, h, gamma)
            w = w - gamma_actual * stoch_gradient
        else:
            w = w - gamma * stoch_gradient
        # if n_iter%20==0:
        #     print("Gradient Descent({bi}/{ti}): loss ={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=np.round(loss,4), w0=np.round(w[0],4), w1=np.round(w[1],4)))
        if plot==True:
            losses.append(loss)
    if plot==True:
        plt.plot(losses, '-', label="AdaGrad method")
        plt.xlabel("Number of iterations")
        plt.ylabel("Average Test loss over the k folds for the best degree")
        plt.legend
        plt.show
    return w, loss

def least_squares(y, tx):
    """Calculates the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss = compute_loss_MSE(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Calculates the least squares solution with ridge constrain."""
    a = tx.T.dot(tx) + 2 * y.size * lambda_ * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w_star = np.linalg.solve(a,b)
    loss = compute_loss_MSE(y, tx, w_star)
    return w_star, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma, param=None):
    ''' Logistic regression using gradient descent or SGD '''
    return reg_logistic_regression(y, tx, 0.0, initial_w, max_iters, gamma, param)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, param=None):
    ''' Regularized logistic regression using gradient descent or SGD '''
    if param is None: param = {}
    if param.get('Decreasing_gamma', None) is None: param.update({'Decreasing_gamma': False})
    if param.get('Decreasing_gamma_final', None) is None: param.update({'Decreasing_gamma_final': 1e-6})
    if param.get('AdaGrad', None) is None: param.update({'AdaGrad': False})
    if param.get('Newton_method', None) is None: param.update({'Newton_method': False})
    if param.get('Batch_size', None) is None: param.update({'Batch_size': 1})
    if param.get('Print_update', None) is None: param.update({'Print_update': False})
    if param['Decreasing_gamma']:
        nb_gamma_update = math.floor(max_iters/100)
        gamma_mult_coeff = math.exp(math.log(param['Decreasing_gamma_final']/gamma)/nb_gamma_update)
    y = y.reshape(-1, 1)
    tx = tx.reshape(tx.shape[0], -1)
    w = initial_w.reshape(-1, 1)
    h = np.zeros(w.shape)
    initial_gamma = gamma
    iter = 0
    stop_iter = False
    while iter < max_iters and stop_iter == False:
        batch_size = param['Batch_size']
        n_batch = math.floor(y.shape[0]/batch_size)
        for minibatch_y, minibatch_tX in batch_iter(y, tx, batch_size, n_batch):
            if iter >= max_iters:
                stop_iter = True
                break
            loss = compute_loss_NLL(minibatch_y, minibatch_tX, w, lambda_)
            grad = compute_gradient_NLL(minibatch_y, minibatch_tX, w, lambda_)
            if param['Decreasing_gamma'] and iter % 100 == 0 and iter != 0:
                gamma = gamma_mult_coeff*gamma
            if param['AdaGrad']:
                gamma, h = ada_grad(grad, h, initial_gamma)
            if param['Newton_method']:
                H = compute_hessian_NLL(minibatch_y, minibatch_tX, w, lambda_)
                w = w - gamma*np.linalg.pinv(H)@grad
            else:
                w = w - gamma*grad
            if param['Print_update'] and iter % 1000 == 0:
                print(f"Current iteration={iter}, loss={loss}")
            iter += 1
    loss = compute_loss_NLL(y, tx, w, lambda_)
    if param['Print_update']: print(f"Final loss={loss}")
    return w, loss

