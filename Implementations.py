import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import*

""""""""""""""""""""""""
"  REQUIRED FUNCTIONS  "
""""""""""""""""""""""""

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm using MSE loss."""
    # Define initial loss and weights
    w = initial_w
    loss = compute_loss_MSE(y, tx, w)
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
    w = initial_w
    loss = compute_loss_MSE(y, tx, w)
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

def logistic_regression():
    return None

def reg_logistic_regression():
    return None


""""""""""""""""""""""""
" ADDITIONAL FUNCTIONS "
""""""""""""""""""""""""

def compute_loss_MSE(y, tX, w):
    """Computes MSE"""
    e = y.reshape(-1,1) - tX@(w.reshape(-1, 1))
    loss_MSE = (e.T@e).item()/(2*y.size)
    return loss_MSE

def compute_loss_NLL(y, tx, w):
    """compute the loss: negative log likelihood."""
    sig = sigmoid(tx.dot(w))
    loss = - np.sum( y*np.log(sig)  +  (1 - y) * np.log(1-sig) )
    return loss

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

def preprocess_data(y, tX, ids, mean=None, std=None, param=None):
    '''
    Preprocessing of the data.
    :param y: labels [n_samples]
    :param tX: data [n_samples x n_dim]
    :param ids: ids of samples [n_samples]
    :param mean: if provided, mean used to standardize the data [optional: float]
    :param std: if provided, std used to standardize the data [optional: float]
    :param param: dict of different parameters to preprocess the data [dict]
                  default: {'Print_info': False, 'Remove_missing': False, 'Standardization': True,
                            'Missing_to_0': True, 'Missing_to_median': False, 'Build_poly': True,
                            'Degree_poly': 9}
    :return: data preprocessed (y, tX, ids, tX_mean, tX_std)
    '''
    if param is None: param = {}
    if param.get('Print_info', None) is None: param.update({'Print_info': False})  # print informations about the data tX
    if param.get('Remove_missing', None) is None: param.update({'Remove_missing': False})  # remove the samples with -999 values
    if param.get('Standardization', None) is None: param.update({'Standardization': True})  # standardize the data
    if param.get('Missing_to_0', None) is None: param.update({'Missing_to_0': True})  # change -999 values to 0.0
    if param.get('Missing_to_median', None) is None: param.update({'Missing_to_median': False})  # change -999 values to the median of their features
    if param.get('Build_poly', None) is None: param.update({'Build_poly': True})  # build polynomial data
    if param.get('Degree_poly', None) is None: param.update({'Degree_poly': 9})  # max degree computed when building polynomial data
    if tX.ndim == 1:
        tX = tX.reshape((-1,1))

    mat_missing = np.full(tX.shape, False)  # matrix [n_samples x n_dim] containing True when tX==-999 and False when tX!=-999
    mat_missing[np.where(tX == -999)] = True
    id_good = np.where(tX.min(axis=1) == -999.0, False, True)  # ids of samples without -999 values

    if param['Print_info']:
        print(f"Minimum value of X: {tX.min()}\nMaximum value of X: {tX.max()}")
        values, counts = np.unique(tX, return_counts=True)
        print(f"Number of -999.0 in X: {dict(zip(values, counts)).get(-999.0,0)}")
        N_good = np.count_nonzero(id_good)
        print(f"Number of samples without -999.0: {N_good}/{id_good.size}")
    if param['Remove_missing']:
        tX = tX[id_good]
        y = y[id_good]
        ids = ids[id_good]
    if param['Standardization']:
        tX_mean = mean
        tX_std = std
        if int(np.__version__.split('.')[0])>=1 and int(np.__version__.split('.')[1])>=20:
            if mean is None: tX_mean = tX.mean(axis=0, where=np.invert(mat_missing)).reshape(1,-1)
            if std is None: tX_std = tX.std(axis=0, where=np.invert(mat_missing)).reshape(1,-1)
        else:
            tX2 = tX.copy()
            tX2[mat_missing] = np.nan
            if mean is None: tX_mean = np.nanmean(tX2, axis=0).reshape(1,-1)
            if std is None: tX_std = np.nanstd(tX2, axis=0).reshape(1,-1)
        tX = (tX-tX_mean)/tX_std
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
    return y, tX, ids, tX_mean, tX_std

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

#def cross_validation(y, x, k_indices, k, lambda_, degree, model):
    #separate line of index taken for test split
#    train_folds = list(range(k_indices.shape[0]))
#    train_folds.remove(k)
#    train_idx = np.concatenate(([k_indices[fold,:] for fold in train_folds]))
#    test_idx = k_indices[k,:]
    # ***************************************************
#    feat_matrix_tr = build_poly(x[train_idx], degree)
#    feat_matrix_te = build_poly(x[test_idx], degree)
#    y_tr = y[train_idx]
#    y_te = y[test_idx]
    # ***************************************************
#    weights, loss_tr = model(y_tr, feat_matrix_tr, lambda_)
#    loss_te = compute_mse(y_te, feat_matrix_te, weights)
#    return loss_tr, loss_te

def cross_validation(y, x, k_indices, k, model, degree = 1, params = None):
    """
    Function used to get training/test loss and accuracy on the kth fold during cross-validation,
    for specific parameter values, for a given model
    :param y: labels
    :param x: data
    :param k_indices: array of indices contained by each fold
    :param k: fold number used as test set
    :param model: model name
    :param degree: degree up to which each parameter will get extended features
    :param params: dictionnary containing parameters relevant among {max_iters, gamma, batch_size, lambda} for the chosen model
    """
    #Recap of the arguments entered as the function is heavy in parameters
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
        max_iters, gamma = params['max_iters'], params['gamma']
        initial_w=np.zeros(feat_matrix_tr.shape[1])
        w, loss_tr = least_squares_GD(y_tr, feat_matrix_tr, initial_w, max_iters, gamma)
    
    elif model == 'least_squares_SGD':
        max_iters, gamma, batch_size = params['max_iters'], params['gamma'], params['batch_size']
        initial_w=np.zeros(feat_matrix_tr.shape[1])
        w, loss_tr = least_squares_SGD(y_tr, feat_matrix_tr, initial_w, max_iters, gamma, batch_size)
    
    elif model == 'ridge_regression':
        lambda_ = params['lambda']
        w, loss_tr = ridge_regression(y_tr, feat_matrix_tr, lambda_)
    
    elif model == 'logistic_regression':
        max_iters, gamma = params['max_iters'], params['gamma']
        initial_w=np.zeros(feat_matrix_tr.shape[1])
        w, loss_tr = logistic_regression(y_tr, feat_matrix_tr, initial_w, max_iters, gamma)
        
    
    elif model == 'ref_logistic_regression':
        lambda_, max_iters, gamma = params['lambda'], params['max_iters'], params['gamma']
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

def params_optimization(y, x, k_fold, model, degrees, lambdas = None, params = None, seed = 1): #gammas, max_iters, batch_size, lambda
    '''
    Optimization of parameters degree and possibly lambda
    :param y: labels [n_samples]
    :param x: data [n_samples x n_dim]
    :param degrees: a list of degrees to test for the best model
    :param lambdas: a list of ridge regularizer to test for the best model
    :param k_fold: number of folds for cross validation
    :param model: model chosen, either 'least_squares_GD' or 'least_squares_SGD'
    :param params: dictionnary of aditional parameters necessary for the model chosen
    :param seed: fixed seed for code reproducibility, by default, 1
    :return: if lambda also optimized, 2 matrix of accuracies (train, test) (degree x lambda) for each degree-lambda combination
             if only degree optimized, 2 vectors of accuracies (train, test) (dim = degree) for each degree
    '''
    #split data in k_fold:
    k_indices = build_k_indices(y, k_fold, seed)
    accs_tr = []
    accs_te = []
    
    if lambdas is None:          
    # Get a mean accuracy value for each degree only
        for degree in degrees:
                degree_accs_tr = []
                degree_accs_te = []
                for k in range(k_fold):
                    _, __, acc_tr, acc_te = cross_validation(y, x, k_indices, k, model, degree, params)
                    degree_accs_tr.append(acc_tr)
                    degree_accs_te.append(acc_te)
                accs_tr.append(np.mean(lambda_accs_tr))
                accs_te.append(np.mean(lambda_accs_te))
    else:
    # Get a mean accuracy value (by cross-validation) for each degree-lambda combination
        for degree in degrees:
            degrees_accs_tr = []
            degrees_accs_te = []
            for lambda_ in lambdas:
                lambda_accs_tr = []
                lambda_accs_te = []
                # Adapt dictionary for cross_validation function
                if params is None:
                    dict_ = {'lambda' : lambda_}
                    params = dict_
                else: params['lambda'] = lambda_
                #start cross-validating on degree-lambda pair
                for k in range(k_fold):
                    _, __, acc_tr, acc_te = cross_validation(y, x, k_indices, k, model, degree, params)
                    lambda_accs_tr.append(acc_tr)
                    lambda_accs_te.append(acc_te)
                degrees_accs_tr.append(np.mean(lambda_accs_tr))
                degrees_accs_te.append(np.mean(lambda_accs_te))
            accs_tr.append(degrees_accs_tr)
            accs_te.append(degrees_accs_te)
              
    return np.array(accs_tr), np.array(accs_te)
              
"""Functions used to plot losses and parameters for model selection"""                                                                     

def plot_param_vs_loss(params, err_tr, err_te, param = 'degree', err_type = 'MSE', model_name = 'model'):
    """
    Visualization of the curves of mse/accuracy given parameter (degree, learning rate, lambda).
     :param params: list of the parameters used for each version of the model
     :param err_tr: corresponding training error, whether mse or accuracy
     :param err_te: corresponding test error, whether mse or accuracy
     :param param: label of the parameter used
     :param err_type: type of error (mse or accuracy)
     :param model_name: name of the model used
    """
    best_idx = np.argmin(err_te)
    
    if param == 'lambda':
        plt.semilogx(params, err_tr, marker=".", color='b', label='train error')
        plt.semilogx(params, err_te, marker=".", color='r', label='test error')
    else:
        plt.plot(params, err_tr, marker=".", color='b', label='train error')
        plt.plot(params, err_te, marker=".", color='r', label='test error')
    plt.axvline(params[best_idx], color = 'k', ls = '--', alpha = 0.5, label = 'best ' + param)
    plt.xlabel(param)
    plt.ylabel(err_type)
    plt.title(err_type + ' of ' + model_name + ' given different values for parameter: ' + param)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_param_vs_losses(params, mse_tr, mse_te, acc_tr, acc_te, param = 'degree', model_name = 'model'):
    """
    Visualization of the curves of mse AND accuracy given parameter (degree, learning rate, lambda).
     :param params: list of the parameters used for each version of the model
     :param mse_tr: corresponding training mse
     :param mse_te: corresponding test mse
     :param acc_tr: corresponding training accuracy
     :param acc_te: corresponding test accuracy
     :param param: label of the parameter used
     :param model_name: name of the model used
    """
    
    best_idx_mse = np.argmin(mse_te)
    best_idx_acc = np.argmin(acc_te)
    
    fig, axs = plt.subplots(1, 2, figsize = [12,5])
    fig.suptitle('MSE and accuracy of ' + model_name + ' given different values for parameter: ' + param)
    if param == 'lambda':
        axs[0].semilogx(params, mse_tr, marker=".", color='b', label='train error')
        axs[0].semilogx(params, mse_te, marker=".", color='r', label='test error')
        axs[1].semilogx(params, acc_tr, marker=".", color='b')
        axs[1].semilogx(params, acc_te, marker=".", color='r')    
    else:
        axs[0].plot(params, mse_tr, marker=".", color='b', label='train error')
        axs[0].plot(params, mse_te, marker=".", color='r', label='test error')
        axs[1].plot(params, acc_tr, marker=".", color='b')
        axs[1].plot(params, acc_te, marker=".", color='r')
    axs[0].axvline(params[best_idx_mse], color = 'k', ls = '--', alpha = 0.5, label = 'best ' + param)
    axs[1].axvline(params[best_idx_acc], color = 'k', ls = '--', alpha = 0.5)
    
    axs[0].set_xlabel(param)
    axs[1].set_xlabel(param)
    axs[0].set_ylabel('MSE')
    axs[1].set_ylabel('Accuracy')
    axs[0].grid(True)
    axs[1].grid(True)
    fig.legend()
    plt.show()
    
def plot_boxplot(losses, model_names, err_type = 'MSE'):
    """Visualisation of the performance of models across folds.
     :param losses: array of losses, such that each ROW contains the losses of a same model on different folds (cross-validation)
     :param model_names: names of the models corresponding to each row
     :param err_type: type of error (mse or accuracy)
    """
    losses = losses.T
    bp = plt.boxplot(losses, labels = model_names, showmeans = True)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
    plt.title('Boxplot of the ' + err_type + ' models (' + str(np.array(losses).shape[1]) + ' folds)')
    plt.ylabel(err_type)
    plt.show()
    
def plot_twice_boxplot(MSEs, accuracies, model_names):
    """Visualisation of the performance of models across folds.
     :param MSEs: array of MSEs. Each ROW contains the MSE of a same model on different folds (cross-validation)
     :param accuracies: array of accuraciess. Each ROW contains the accuracy of a same model on different folds (cross-validation)
     :param model_names: names of the models corresponding to each row
    """
    MSEs = MSEs.T
    accuracies = accuracies.T
    fig, axs = plt.subplots(1, 2, figsize = [12,5])
    fig.suptitle('Boxplot of the MSE and accuracy of models (' + str(np.array(MSEs).shape[1]) + ' folds)')
    axs[0].boxplot(MSEs, labels = model_names, showmeans = True)
    axs[0].set_ylabel('MSE')
    bp = axs[1].boxplot(accuracies, labels = model_names, showmeans = True)
    axs[1].set_ylabel('Accuracy')
    fig.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
    plt.show()
    
def plot_heatmap(losses_tr, losses_te, degrees, lambdas, model_name):
    fig, axs = plt.subplots(1, 2, figsize = [12,5])
    fig.suptitle('Accuracy of ' + model_name + ' given different values for parameter lambda and degree.')
    im0 = axs[0].imshow(losses_tr)
    im1 = axs[1].imshow(losses_te)
    

    # We want to show all ticks...
    axs[0].set_xticks(np.arange(len(lambdas)))
    axs[0].set_yticks(np.arange(len(degrees)))
    # ... and label them with the respective list entries
    axs[0].set_xticklabels(lambdas)
    axs[0].set_yticklabels(degrees)
    
    axs[1].set_xticks(np.arange(len(lambdas)))
    axs[1].set_yticks(np.arange(len(degrees)))
    # ... and label them with the respective list entries
    axs[1].set_xticklabels(lambdas)
    axs[1].set_yticklabels(degrees)

    # Rotate the tick labels and set their alignment.
    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(degrees)):
        for j in range(len(lambdas)):
            text = axs[0].text(j, i, losses_tr[i, j],
                           ha="center", va="center", color="w")
            text = axs[1].text(j, i, losses_te[i, j],
                           ha="center", va="center", color="w")

    axs[0].set_title("Train accuracy")
    axs[1].set_title("Test accuracy")
    fig.tight_layout()
    plt.show()