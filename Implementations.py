import math
import warnings
import numpy as np
from Secondary import *
import matplotlib.pyplot as plt

"""""""""""""""""""""""
" REQUIRED FUNCTIONS  "
"""""""""""""""""""""""

def least_squares_GD(y, tx, initial_w, max_iters, gamma, plot=False, adagrad=False):
    '''
    Gradient descent algorithm using MSE loss.
    :param y: labels
    :param tx: data
    :param initial_w: weights
    :param max_iters: total number of iterations
    :param gamma: learning rate
    :param plot: plot the training loss
    :param adagrad: use AdaGrad implementation of the learning rate
    :return: optimized weights, final loss
    '''
    # Define initial loss and weights
    w = initial_w.reshape(-1, 1)
    h = np.zeros(w.shape)
    if plot:
        loss = compute_loss_MSE(y, tx, w)
        losses = [loss]
        iters = [0]
    for n_iter in range(max_iters):
        gradient = compute_gradient_MSE(y, tx, w)
        if adagrad:
            gamma_actual, h = ada_grad(gradient, h, gamma)
            w = w - gamma_actual * gradient
        else:
            w = w - gamma * gradient
        if plot and n_iter % 10 == 0:
            loss = compute_loss_MSE(y, tx, w)
            losses.append(loss)
            iters.append(n_iter+1)
            # print("Gradient Descent({bi}/{ti}): loss ={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=np.round(loss,4), w0=np.round(w[0],4), w1=np.round(w[1],4)))
    if plot:
        plt.plot(iters, losses, '-', label="AdaGrad method")
        plt.xlabel("Number of iterations")
        plt.ylabel("Average Train loss over the k folds for the best degree")
        plt.legend
        plt.show
    loss = compute_loss_MSE(y, tx, w)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size = 1, plot=False, adagrad=False):
    '''
    Stochastic gradient descent algorithm using MSE loss.
    :param y: labels
    :param tx: data
    :param initial_w: weights
    :param max_iters: total number of iterations
    :param gamma: learning rate
    :param batch_size: batch size, default is 1 (SGD)
    :param plot: plot the training loss
    :param adagrad: use AdaGrad implementation of the learning rate
    :return: optimized weights, final loss
    '''
    # Define parameters to store w and loss
    w = initial_w.reshape(-1, 1)
    h = np.zeros(w.shape)
    if plot:
        loss = compute_loss_MSE(y, tx, w)
        losses = [loss]
        iters = [0]
    n_iter = 0
    stop_iter = False
    while n_iter < max_iters and stop_iter == False:
        n_batch = math.floor(y.shape[0] / batch_size)
        for minibatch_y, minibatch_tX in batch_iter(y, tx, batch_size, n_batch):
            if n_iter >= max_iters:
                stop_iter = True
                break
            stoch_gradient = compute_gradient_MSE(minibatch_y, minibatch_tX, w)
            if adagrad:
                gamma_actual, h=ada_grad(stoch_gradient, h, gamma)
                w = w - gamma_actual * stoch_gradient
            else:
                w = w - gamma * stoch_gradient
            if plot and n_iter % 10 == 0:
                loss = compute_loss_MSE(y, tx, w)
                losses.append(loss)
                iters.append(n_iter+1)
                # print("Gradient Descent({bi}/{ti}): loss ={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=np.round(loss,4), w0=np.round(w[0],4), w1=np.round(w[1],4)))
            n_iter += 1
    if plot:
        plt.plot(iters, losses, '-', label="AdaGrad method")
        plt.xlabel("Number of iterations")
        plt.ylabel("Average Train loss over the k folds for the best degree")
        plt.legend
        plt.show
    loss = compute_loss_MSE(y, tx, w)
    return w, loss

def least_squares(y, tx):
    '''
    Calculates the least squares solution.
    :param y: labels
    :param tx: data
    :return: optimized weights, final loss
    '''
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss = compute_loss_MSE(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    '''
    Calculates the least squares solution with ridge constrain.
    :param y: labels
    :param tx: data
    :param lambda_: regularization term (lambda)
    :return: optimized weights, final loss
    '''
    a = tx.T.dot(tx) + 2 * y.size * lambda_ * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w_star = np.linalg.solve(a,b)
    loss = compute_loss_MSE(y, tx, w_star)
    return w_star, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma, param=None):
    '''
    Logistic regression using gradient descent or SGD
    :param y: labels
    :param tx: data
    :param initial_w: weights
    :param max_iters: total number of iterations
    :param gamma: learning rate
    :param param: additional parameters
              default: {'Decreasing_gamma': False      | Use decreasing gamma as learning rate
                        'Decreasing_gamma_final': 1e-6 | Final learning rate for decreasing gamma
                        'AdaGrad': False               | Use AdaGrad as learning rate
                        'Newton_method': False         | Use Newton method to upgrade the weights
                        'Batch_size': 1                | Batch size used (default is 1: SGD)
                        'Print_update': False}         | Print the update informations
    :return: optimized weights, final loss
    '''
    return reg_logistic_regression(y, tx, 0.0, initial_w, max_iters, gamma, param)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, param=None):
    '''
    Regularized logistic regression using gradient descent or SGD.
    :param y: labels
    :param tx: data
    :param lambda_: regularization term (lambda)
    :param initial_w: weights
    :param max_iters: total number of iterations
    :param gamma: learning rate
    :param param: additional parameters
                  default: {'Decreasing_gamma': False      | Use decreasing gamma as learning rate
                            'Decreasing_gamma_final': 1e-6 | Final learning rate for decreasing gamma
                            'AdaGrad': False               | Use AdaGrad as learning rate
                            'Newton_method': False         | Use Newton method to upgrade the weights
                            'Batch_size': 1                | Batch size used (default is 1: SGD)
                            'Print_update': False}         | Print the update informations
    :return: optimized weights, final loss
    '''
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
    def check_y(y):
        if np.any(y == 0):
            warnings.warn("This function implements logistic equations for y={-1,1} ! Therefore you need to convert your labels to y={-1,1}.")
        y[y == 0] = -1
        return y
    y = y.reshape(-1, 1)
    y = check_y(y)
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
            if param['Print_update'] and iter % 100 == 0:
                loss = compute_loss_NLL(y, tx, w, lambda_)
                print(f"Current iteration={iter}, loss={loss}")
            iter += 1
    loss = compute_loss_NLL(y, tx, w, lambda_)
    if param['Print_update']: print(f"Final loss={loss}")
    return w, loss

""""""""""""""""""""""""
" ADDITIONAL FUNCTIONS "
""""""""""""""""""""""""

def build_k_indices(y, k_fold, seed):
    '''
    Building k indices for k_fold cross validation.
    :param y: labels [n_samples]
    :param k_fold: number of folds for cross validation
    :param seed: fixed seed for reproducibility of the data (random permutations for indices)
    :return: an array of k_indices
    '''
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    # print(interval)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    # print(indices)
    k_indices = [indices[k * interval: (k + 1) * interval]  # end pas inclus
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, model, degree=1, params=None, params_logistic=None, feedback=False, x_poly_built=False, x_poly_deg=16):
    """
    Function used to get training/test loss and accuracy on the kth fold during cross-validation,
    for specific parameter values, for a given model
    :param y: labels
    :param x: data
    :param k_indices: array of indices contained by each fold
    :param k: fold number used as test set
    :param model: model name
    :param degree: degree up to which each parameter will get extended features
    :param params: dictionnary containing parameters relevant among {max_iters, gamma_zero, batch_size, lambda} for the chosen model
    :param params_logistic: special parameters controlling the gradient descent process, used by logistic regression functions
    :param feedback: enables feedback of cross-validation progression
    :param x_poly_built: True if x already contains polynomial expansion
    :param x_poly_deg: degree of the already computed polynomial expansion of x (if x_poly_built=True)
    """
    # Recap of the arguments entered as the function is heavy in parameters
    if feedback:
        print('Starting cross-validation {}/{} for {}, extended feature of degree {} and arguments : {}'.format(k + 1,len(k_indices),model,degree,params))

    # Create k-th split of train/test sets, possibly with extended features
    train_folds = list(range(k_indices.shape[0]))
    train_folds.remove(k)
    train_idx = np.concatenate(([k_indices[fold, :] for fold in train_folds]))
    test_idx = k_indices[k, :]

    if x_poly_built == False:
        feat_matrix_tr = build_poly(x[train_idx], degree)
        feat_matrix_te = build_poly(x[test_idx], degree)
    else:
        new_x = x[:,0].reshape(-1,1)
        for d in range(1, degree+1):
            new_x = np.hstack((new_x, x[:,d::x_poly_deg]))
        feat_matrix_tr = new_x[train_idx]
        feat_matrix_te = new_x[test_idx]

    y_tr = y[train_idx]
    y_te = y[test_idx]
    initial_w = np.random.normal(0.0, 0.1, size=(feat_matrix_tr.shape[1], 1))

    # Use model given in parameter and initialize relevant parameters
    if model == 'least_squares':
        w, loss_tr = least_squares(y_tr, feat_matrix_tr)
    elif model == 'least_squares_GD':
        max_iters, plot = params['max_iters'], params['plot']
        gamma_zero = 0.1 * np.ones(feat_matrix_tr.shape[1])
        w, loss_tr = least_squares_GD(y_tr, feat_matrix_tr, initial_w, max_iters, gamma_zero, plot, adagrad=True)
    elif model == 'least_squares_SGD':
        max_iters, batch_size, plot = params['max_iters'], params['batch_size'], params['plot']
        gamma_zero = 0.01 * np.ones(feat_matrix_tr.shape[1])
        w, loss_tr = least_squares_SGD(y_tr, feat_matrix_tr, initial_w, max_iters, gamma_zero, batch_size, plot, adagrad=True)
    elif model == 'ridge_regression':
        lambda_ = params['lambda']
        w, loss_tr = ridge_regression(y_tr, feat_matrix_tr, lambda_)
    elif model == 'logistic_regression':
        max_iters, gamma = params['max_iters'], params['gamma']
        w, loss_tr = logistic_regression(y_tr, feat_matrix_tr, initial_w, max_iters, gamma, param=params_logistic)
    elif model == 'reg_logistic_regression':
        lambda_, max_iters, gamma = params['lambda'], params['max_iters'], params['gamma']
        w, loss_tr = reg_logistic_regression(y_tr, feat_matrix_tr, lambda_, initial_w, max_iters, gamma, param=params_logistic)
    else:
        print('Model choice incorrect, execution halted.')
        exit()

    # MSE returned for comparison purposes, but accuracy is used as model evaluator
    if model == 'logistic_regression' or model == 'reg_logistic_regression':
        loss_te = compute_loss_NLL(y_te, feat_matrix_te, w)
    else:
        loss_te = compute_loss_MSE(y_te, feat_matrix_te, w)

    acc_tr = compute_accuracy(y_tr, feat_matrix_tr, w)
    acc_te = compute_accuracy(y_te, feat_matrix_te, w)
    return loss_tr, loss_te, acc_tr, acc_te

def params_optimization(y, x, k_fold, model, degrees, lambdas=None, params=None, params_logistic=None, seed=1, feedback=False):
    '''
    Optimization of parameters degree and possibly lambda
    :param y: labels [n_samples]
    :param x: data [n_samples x n_dim]
    :param k_fold: number of folds for cross validation
    :param model: model chosen, either 'least_squares_GD' or 'least_squares_SGD'
    :param degrees: a list of degrees to test for the best model
    :param lambdas: a list of ridge regularizer to test for the best model
    :param max_iters: number of iterations for linear or logistic regression using gradient descent, stochastic gradient descent
    :param batch_size: the size of the batch if using batch gradient descent
    :param params: dictionnary of aditional parameters necessary for the model chosen
    :param params_logistic: special parameters controlling the gradient descent process, used by logistic regression functions
    :param seed: fixed seed for code reproducibility, by default, 1
    :param feedback: enables feedback of cross-validation progression
    :return: if lambda also optimized, 4 matrix of loss and accuracies (train, test) (degree x lambda) for each degree-lambda combination
             if only degree optimized, 4 vectors of loss and accuracies (train, test) (dim = degree) for each degree
    '''
    # split data in k_fold:
    k_indices = build_k_indices(y, k_fold, seed)
    accs_tr = []
    accs_te = []
    losses_tr = []
    losses_te = []

    if type(degrees) is list:
        max_degree = max(degrees)
    elif type(degrees) is np.ndarray:
        max_degree = np.max(degrees)
    else:
        max_degree = 20
    x = build_poly(x, max_degree)

    if lambdas is None:
        # Get a mean accuracy value for each degree only
        for degree in degrees:
            degree_accs_tr = []
            degree_accs_te = []
            degree_losses_tr = []
            degree_losses_te = []
            # give feedback
            if feedback:
                print('Optimizing degree {}/{}, model: {}, arguments: {}'.format(degree, np.array(degrees)[-1], model,params))
            for k in range(k_fold):
                loss_tr, loss_te, acc_tr, acc_te = cross_validation(y, x, k_indices, k, model, degree, params,params_logistic=params_logistic, x_poly_built=True, x_poly_deg=max_degree)
                degree_accs_tr.append(acc_tr)
                degree_accs_te.append(acc_te)
                degree_losses_tr.append(loss_tr)
                degree_losses_te.append(loss_te)
            accs_tr.append(np.mean(degree_accs_tr))
            accs_te.append(np.mean(degree_accs_te))
            losses_tr.append(np.mean(degree_losses_tr))
            losses_te.append(np.mean(degree_losses_te))
    else:
        # Get a mean accuracy value (by cross-validation) for each degree-lambda combination
        for degree in degrees:
            degree_accs_tr = []
            degree_accs_te = []
            degree_losses_tr = []
            degree_losses_te = []
            for lambda_ in lambdas:
                lambda_accs_tr = []
                lambda_accs_te = []
                lambda_losses_tr = []
                lambda_losses_te = []
                # Adapt dictionary for cross_validation function
                if params is None:
                    dict_ = {'lambda': lambda_}
                    params = dict_
                else:
                    params['lambda'] = lambda_
                # give feedback
                if feedback:
                    print('Optimizing degree {}/{}, model: {}, arguments: {}'.format(degree, np.array(degrees)[-1], model,
                                                                                   params))
                # start cross-validating on degree-lambda pair
                for k in range(k_fold):
                    loss_tr, loss_te, acc_tr, acc_te = cross_validation(y, x, k_indices, k, model, degree, params,params_logistic=params_logistic, x_poly_built=True, x_poly_deg=max_degree)
                    lambda_accs_tr.append(acc_tr)
                    lambda_accs_te.append(acc_te)
                    lambda_losses_tr.append(loss_tr)
                    lambda_losses_te.append(loss_te)
                degree_accs_tr.append(np.mean(lambda_accs_tr))
                degree_accs_te.append(np.mean(lambda_accs_te))
                degree_losses_tr.append(np.mean(lambda_losses_tr))
                degree_losses_te.append(np.mean(lambda_losses_te))
            accs_tr.append(degree_accs_tr)
            accs_te.append(degree_accs_te)
            losses_tr.append(degree_losses_tr)
            losses_te.append(degree_losses_te)

    return np.array(losses_tr), np.array(losses_te), np.array(accs_tr), np.array(accs_te)