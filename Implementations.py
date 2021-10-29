import math
import numpy as np
from Secondary import *
import matplotlib.pyplot as plt

""""""""""""""""""""""""
"  REQUIRED FUNCTIONS  "
""""""""""""""""""""""""

def least_squares_GD(y, tx, initial_w, max_iters, gamma_zero, plot=False):
    """Gradient descent algorithm using MSE loss."""
    # Define initial loss and weights
    h=np.zeros(tx.shape[1])
    w = initial_w
    loss = compute_loss_MSE(y, tx, w)
    losses=[]
    for n_iter in range(max_iters):                                    
        loss = compute_loss_MSE(y, tx, w)
        gradient = compute_gradient_MSE(y, tx, w)
        gamma, h=ada_grad(gradient, h, gamma_zero)
        w = w - gamma * gradient
        if n_iter%20==0:
            print("Gradient Descent({bi}/{ti}): loss ={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=np.round(loss,4), w0=np.round(w[0],4), w1=np.round(w[1],4)))
        if plot:
            losses.append(loss)
    if plot:
        plt.plot(losses, '-', label="AdaGrad method")
        plt.xlabel("Number of iterations")
        plt.ylabel("Average Test loss over the k folds for the best degree")
        plt.legend
        plt.show
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma_zero, batch_size = 1, plot=False):
    """Stochastic gradient descent algorithm using MSE loss."""
    # Define parameters to store w and loss
    h=np.zeros(tx.shape[1])
    w = initial_w
    loss = compute_loss_MSE(y, tx, w)
    losses=[]
    for n_iter in range(max_iters):
        generator = batch_iter(y, tx, batch_size)                      
        y_sub, tx_sub = next(generator)
        loss = compute_loss_MSE(y_sub, tx_sub, w)
        stoch_gradient = compute_gradient_MSE(y_sub, tx_sub, w)
        gamma, h=ada_grad(stoch_gradient, h, gamma_zero)
        w = w - gamma * stoch_gradient
        if n_iter%20==0:
            print("Gradient Descent({bi}/{ti}): loss ={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=np.round(loss,4), w0=np.round(w[0],4), w1=np.round(w[1],4)))
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
    a = tx.T.dot(tx) + 2*len(y)*lambda_*np.eye(len(tx.T))
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss = compute_loss_MSE(y, tx, w)
    return w, loss

def logistic_regression(y, tX, initial_w, max_iters, gamma, param=None):
    ''' Logistic regression using gradient descent or SGD '''
    return reg_logistic_regression(y, tX, 0.0, initial_w, max_iters, gamma, param)

def reg_logistic_regression(y, tX, lambda_, initial_w, max_iters, gamma, param=None):
    ''' Regularized logistic regression using gradient descent or SGD '''
    if param is None: param = {}
    if param.get('Decreasing_gamma', None) is None: param.update({'Decreasing_gamma': False})
    if param.get('Decreasing_gamma_final', None) is None: param.update({'Decreasing_gamma_final': 1e-4})
    if param.get('AdaGrad', None) is None: param.update({'AdaGrad': False})
    if param.get('Newton_method', None) is None: param.update({'Newton_method': True})
    if param.get('Batch_size', None) is None: param.update({'Batch_size': 1})
    if param.get('Print_update', None) is None: param.update({'Print_update': False})
    if param['Decreasing_gamma']:
        nb_gamma_update = math.floor(max_iters/100)
        gamma_mult_coeff = math.exp(math.log(param['Decreasing_gamma_final']/gamma)/nb_gamma_update)
    y = y.reshape(-1, 1)
    tX = tX.reshape(tX.shape[0], -1)
    w = initial_w.reshape(-1, 1)
    losses = []
    h = np.zeros(w.shape)
    initial_gamma = gamma
    iter = 0
    stop_iter = False
    while iter < max_iters and stop_iter == False:
        batch_size = param['Batch_size']
        n_batch = math.floor(y.shape[0]/batch_size)
        for minibatch_y, minibatch_tX in batch_iter(y, tX, batch_size, n_batch):
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
            losses.append(loss)
            if param['Print_update'] and iter % 1000 == 0:
                print(f"Current iteration={iter}, loss={loss}")
            iter += 1
    loss = compute_loss_NLL(y, tX, w, lambda_)
    if param['Print_update']: print(f"Final loss={loss}")
    return w, loss


""""""""""""""""""""""""
" ADDITIONAL FUNCTIONS "
""""""""""""""""""""""""

def preprocess_data(y_train, tX_train, ids_train, tX_test, ids_test, param=None):
    '''
    Preprocessing of the data.
    :param y_train: labels [n_samples]
    :param tX_train: data [n_samples x n_dim]
    :param ids_train: ids of samples [n_samples]
    :param mean: if provided, mean used to standardize the data [optional: float]
    :param std: if provided, std used to standardize the data [optional: float]
    :param param: dict of different parameters to preprocess the data [dict]
                  default: {'Print_info': False, 'Remove_missing': False, 'Remove_random_parameters': True,
                            'Standardization': True, 'Missing_to_0': True, 'Missing_to_median': False,
                            'Build_poly': True, 'Degree_poly': 9, 'Standardization_build_poly': False}
    :return: data preprocessed (y, tX, ids, tX_mean, tX_std)
    '''
    if param is None: param = {}
    if param.get('Print_info', None) is None: param.update({'Print_info': False})  # print informations about the data tX
    if param.get('Remove_missing', None) is None: param.update({'Remove_missing': False})  # remove the samples with -999 values
    if param.get('Remove_random_parameters', None) is None: param.update({'Remove_random_parameters': True})  # remove the useless parameters
    if param.get('Standardization', None) is None: param.update({'Standardization': True})  # standardize the data
    if param.get('Missing_to_0', None) is None: param.update({'Missing_to_0': True})  # change -999 values to 0.0
    if param.get('Missing_to_median', None) is None: param.update({'Missing_to_median': False})  # change -999 values to the median of their features
    if param.get('Remove_outliers', None) is None: param.update({'Remove_outliers': True})
    if param.get('Build_poly', None) is None: param.update({'Build_poly': True})  # build polynomial data
    if param.get('Degree_poly', None) is None: param.update({'Degree_poly': 9})  # max degree computed when building polynomial data
    if param.get('Standardization_build_poly', None) is None: param.update({'Standardization_build_poly': False})
    if tX_train.ndim == 1:
        tX_train = tX_train.reshape((-1, 1))
    if tX_test.ndim == 1:
        tX_test = tX_train.reshape((-1, 1))

    tX = np.vstack((tX_train, tX_test))
    mat_missing = np.full(tX.shape, False)  # matrix [n_samples x n_dim] containing True when tX==-999 and False when tX!=-999
    mat_missing[np.where(tX == -999)] = True
    id_good_train = np.where(tX_train.min(axis=1) == -999.0, False, True)  # ids of samples in X_train without -999 values
    if param['Print_info']:
        print(f"Minimum value of X_train: {tX_train.min()}\nMaximum value of X_train: {tX_train.max()}")
        values, counts = np.unique(tX_train, return_counts=True)
        print(f"Number of -999.0 in X_train: {dict(zip(values, counts)).get(-999.0,0)}")
        N_good = np.count_nonzero(id_good_train)
        print(f"Number of samples without -999.0 in X_train: {N_good}/{id_good_train.size}")
    if param['Remove_missing']:
        tX = np.vstack((tX[:tX_train.shape[0],:][id_good_train],tX[-tX_test.shape[0]:,:]))
        y_train = y_train[id_good_train]
        ids_train = ids_train[id_good_train]
        mat_missing = np.full(tX.shape, False)
        mat_missing[np.where(tX == -999)] = True
    if param['Remove_random_parameters']:
        tX = np.delete(tX, [15,18,20,25,28], axis=1)
        mat_missing = np.full(tX.shape, False)
        mat_missing[np.where(tX == -999)] = True
    if param['Standardization']:
        if int(np.__version__.split('.')[0])>=1 and int(np.__version__.split('.')[1])>=20:
            tX_mean = tX.mean(axis=0, where=np.invert(mat_missing)).reshape(1, -1)
            tX_std = tX.std(axis=0, where=np.invert(mat_missing)).reshape(1, -1)
        else:
            tX2 = tX.copy()
            tX2[mat_missing] = np.nan
            tX_mean = np.nanmean(tX2, axis=0).reshape(1,-1)
            tX_std = np.nanstd(tX2, axis=0).reshape(1,-1)
        if param['Remove_outliers']:
            mat_missing[tX < tX_mean - 3 * tX_std] = True
            mat_missing[tX > tX_mean + 3 * tX_std] = True
        tX = (tX - tX_mean) / tX_std
    if param['Missing_to_0']:
        tX[mat_missing] = 0.0
    if param['Missing_to_median']:
        tX2 = tX.copy()
        tX2[mat_missing] = np.nan
        tX_median = np.nanmedian(tX2, axis=0)  # medians of the parameters [n_dim]
        ind_missing = np.where(mat_missing == True)
        tX[ind_missing] = tX_median[ind_missing[1]]
    if param['Build_poly']:
        tX = build_poly(tX, param['Degree_poly'])
        if param['Standardization_build_poly']:
            mat_missing = np.tile(mat_missing, (1, param['Degree_poly']))
            if int(np.__version__.split('.')[0]) >= 1 and int(np.__version__.split('.')[1]) >= 20:
                tX_mean = tX[:,1:].mean(axis=0, where=np.invert(mat_missing)).reshape(1, -1)
                tX_std = tX[:,1:].std(axis=0, where=np.invert(mat_missing)).reshape(1, -1)
            else:
                tX2 = tX[:,1:].copy()
                tX2[mat_missing] = np.nan
                tX_mean = np.nanmean(tX2, axis=0).reshape(1, -1)
                tX_std = np.nanstd(tX2, axis=0).reshape(1, -1)
            tX[:,1:] = (tX[:,1:] - tX_mean) / tX_std
    tX_train = tX[:-tX_test.shape[0], :]
    tX_test = tX[-tX_test.shape[0]:,:]
    return y_train, tX_train, ids_train, tX_test, ids_test



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
    #print(interval)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    #print(indices)
    k_indices = [indices[k * interval: (k + 1) * interval] #end pas inclus
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, model, degree = 1, params = None, feedback = False):
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
    :param feedback: enables feedback of cross-validation progression
    """
    #Recap of the arguments entered as the function is heavy in parameters
    if feedback:
        print('Starting cross-validation {}/{} for {}, extended feature of degree {} and arguments : {}'.format(k+1, len(k_indices), model, degree, params))
    
    #Create k-th split of train/test sets, possibly with extended features
    train_folds = list(range(k_indices.shape[0]))
    train_folds.remove(k)
    train_idx = np.concatenate(([k_indices[fold,:] for fold in train_folds]))
    test_idx = k_indices[k,:]
    
    feat_matrix_tr = build_poly(x[train_idx], degree)
    feat_matrix_te = build_poly(x[test_idx], degree)
    y_tr = y[train_idx]
    y_te = y[test_idx]
    
    #Use model given in parameter and initialize relevant parameters
    if model == 'least_squares':
        w, loss_tr = least_squares(y_tr, feat_matrix_tr)
        
    elif model == 'least_squares_GD':
        max_iters, plot= params['max_iters'], params['plot']
        initial_w=np.zeros(feat_matrix_tr.shape[1])
        gamma_zero=0.1*np.ones(feat_matrix_tr.shape[1])
        w, loss_tr = least_squares_GD(y_tr, feat_matrix_tr, initial_w, max_iters, gamma_zero, plot)
    
    elif model == 'least_squares_SGD':
        max_iters, batch_size, plot = params['max_iters'], params['batch_size'], params['plot']
        initial_w=np.zeros(feat_matrix_tr.shape[1])
        gamma_zero=0.01*np.ones(feat_matrix_tr.shape[1])
        w, loss_tr = least_squares_SGD(y_tr, feat_matrix_tr, initial_w, max_iters, gamma_zero, batch_size, plot)
    
    elif model == 'ridge_regression':
        lambda_ = params['lambda']
        w, loss_tr = ridge_regression(y_tr, feat_matrix_tr, lambda_)
    
    elif model == 'logistic_regression':
        max_iters, gamma = params['max_iters'], params['gamma']
        initial_w=np.zeros(feat_matrix_tr.shape[1])
        w, loss_tr = logistic_regression(y_tr, feat_matrix_tr, initial_w, max_iters, gamma)
        
    
    elif model == 'reg_logistic_regression':
        lambda_, max_iters, gamma = params['lambda'], params['max_iters'], params['gamma']
        initial_w = np.zeros(feat_matrix_tr.shape[1])
        w, loss_tr = reg_logistic_regression(y_tr, feat_matrix_tr, lambda_, initial_w, max_iters, gamma)
    
    else:
        print('Model choice incorrect, execution halted.')
        exit()
    
    #MSE returned for comparison purposes, but accuracy is used as model evaluator
    if model == 'logistic_regression' or model == 'reg_logistic_regression':
        loss_te = compute_loss_NLL(y_te, feat_matrix_te, w)
    else:
        loss_te = compute_loss_MSE(y_te, feat_matrix_te, w)

    acc_tr = compute_accuracy(y_tr, feat_matrix_tr, w)
    acc_te = compute_accuracy(y_te, feat_matrix_te, w)
    return loss_tr, loss_te, acc_tr, acc_te

# def learning_rate_optimization(y, x, degrees, k_fold, gammas, max_iters, model, seed=1, batch_size=1):
#     '''
#     Optimization of the learning rate gamma among various degrees for GD or SGD
#     :param y: labels [n_samples]
#     :param x: data [n_samples x n_dim]
#     :param degrees: a list of degrees to test for the best model
#     :param k_fold: number of folds for cross validation
#     :param gammas: a list of learning rate to test, usually ranging from 0.1 to 2
#     :param max_iters: the number of iterations
#     :param model: model chosen, either 'least_squares_GD' or 'least_squares_SGD'
#     :param seed: fixed seed for code reproducibility, by default, 1
#     :param batch_size: the size of batch if using batch gradient descent, by default 1
#     :return: the best learning rate, gamma and degree for gradient descent (best_gamma, best_degree, best_rmse)
#     '''
#     #split data in k_fold:
#     k_indices=build_k_indices(y, k_fold, seed)
#     # for each degree, compute the best learning rate and associated rmse:
#     best_gammas = []
#     best_rmse = []
#     # varying degree:
#     for degree in degrees:
#         #and then performing cross-validation:
#         test_rmse=[]
#         for gamma in gammas:
#             test_rmse_tab = []
#             for k in range(k_fold):
#                 _, loss_te = cross_validation(y, x, k_indices, k, 'least_squares_GD', degree, max_iters=max_iters, gamma=gamma, batch_size=batch_size)
#                 test_rmse_tab.append(loss_te)
#             test_rmse.append(np.mean(test_rmse_tab))
#         ind=np.argmin(test_rmse)
#         best_gammas.append(gammas[ind])
#         best_rmse.append(test_rmse[ind])
#     best_degree=np.argmin(best_rmse)
#     return degrees[best_degree], best_gammas[best_degree], best_rmse[best_degree]

def params_optimization(y, x, k_fold, model, degrees, lambdas = None, params = None, seed = 1, feedback = False): #max_iters, batch_size, lambda
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
    :param seed: fixed seed for code reproducibility, by default, 1
    :param feedback: enables feedback of cross-validation progression
    :return: if lambda also optimized, 4 matrix of loss and accuracies (train, test) (degree x lambda) for each degree-lambda combination
             if only degree optimized, 4 vectors of loss and accuracies (train, test) (dim = degree) for each degree
    '''
    #split data in k_fold:
    k_indices = build_k_indices(y, k_fold, seed)
    accs_tr = []
    accs_te = []
    losses_tr = []
    losses_te = []
    #print(lambdas)
    if lambdas is None:          
    # Get a mean accuracy value for each degree only
        for degree in degrees:
            degree_accs_tr = []
            degree_accs_te = []
            degree_losses_tr = []
            degree_losses_te = []
            # Adapt dictionary for cross_validation function
            #if params is None:
                #dict_ = {'max_iters' : max_iters, 'batch_size': batch_size}
                #params = dict_
            #else: params['max_iters'] = max_iters, params['batch_size']= batch_size
            # give feedback
            if feedback:
                print('Optimizing degree {}/{}, model: {}, arguments: {}'.format(degree, np.array(degrees)[-1], model, params))
            for k in range(k_fold):
                loss_tr, loss_te, acc_tr, acc_te = cross_validation(y, x, k_indices, k, model, degree, params)
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
                    dict_ = {'lambda' : lambda_}
                    params = dict_
                else: params['lambda'] = lambda_
                # give feedback
                if feedback:
                    print('Optimizing degree {}/{}, model: {}, arguments: {}'.format(degree, np.array(degrees)[-1], model, params))
                #start cross-validating on degree-lambda pair
                for k in range(k_fold):
                    loss_tr, loss_te, acc_tr, acc_te = cross_validation(y, x, k_indices, k, model, degree, params)
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