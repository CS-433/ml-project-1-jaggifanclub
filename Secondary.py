import numpy as np
from proj1_helpers import *
from Implementations import *

""""""""""""""
" PREPROCESS "
""""""""""""""

def split_data(y, tX, ratio, seed=1):
    '''
    Split the dataset based on the split ratio. If ratio is 0.8 you will have 80% of
    your data set dedicated to training and the rest dedicated to validation
    :param y: labels
    :param tX: data
    :param ratio: ratio [float]
    :param seed: random seed [int]
    :return: tX_train, y_train, tX_validation, y_validation (splited data)
    '''
    #set seed
    np.random.seed(seed)

    if ratio == 1.0: return tX, y, np.array([]), np.array([])
    N = tX.shape[0]
    limit = int(ratio*N)
    ind = np.random.permutation(N)

    # split data
    tX_train = tX[ind[:limit]].reshape(limit,-1)
    tX_validation = tX[ind[limit:]].reshape(N-limit,-1)
    y_train = y[ind[:limit]]
    y_validation = y[ind[limit:]]

    return tX_train, y_train, tX_validation, y_validation

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

def feature_engineering(tX, param=None):
    '''
    Feature engineering of the data.
    :param tX: data [n_samples x n_dim]
    :param param: dict of different parameters to perform feature engineering to the data [dict]
                  for the following explanations, we use 'p' as the number of base parameters (number of columns in tX without the bias column)
                  default: {'Build_poly': False,         | build polynomial expansion [add p*('Build_poly_degree'-1) columns]
                            'Build_poly_degree': 9,      | degree of the polynomial expansion
                            'Build_root': False,         | build root expansion [add p*('Build_root_degree'-1) columns]
                            'Build_root_degree': 3,      | degree of root expansion, if degree=3: sqrt and cbrt are added
                            'Build_xy': False,           | build correlation expansion [add p*(p-1)/2 columns]
                            'Build_inverse': False,      | build inverse expansion [add p columns]
                            'Build_log': False,          | build natural log expansion [add p columns]
                            'Build_sin_cos': False,      | build sin/cos expansion [add 2*p*'Build_sin_cos_degree' columns]
                            'Build_sin_cos_degree': 3,   | degree of sin/cos expansion, if degree=3: sin, cos, sin^2, cos^2, sin^3, cos^3 are added
                            'Build_sinh_cosh': False,    | build sinh/cosh expansion [add 2*p columns]
                            'Build_rbf_gaussian': False, | build gaussian function (one Radial Basis Function) [add p columns]
                            'Build_all': False}          | build all the above expansions (set all paramters to True)
    :return: feature engineered data [n_samples x ...] with the bias column
    '''
    if param is None: param = {}
    param = {}

    if param.get('Build_poly', None) is None: param.update({'Build_poly': False})
    if param.get('Build_poly_degree', None) is None: param.update({'Build_poly_degree': 9})
    if param.get('Build_root', None) is None: param.update({'Build_root': False})
    if param.get('Build_root_degree', None) is None: param.update({'Build_root_degree': 3})
    if param.get('Build_xy', None) is None: param.update({'Build_xy': False})
    if param.get('Build_inverse', None) is None: param.update({'Build_inverse': False})
    if param.get('Build_log', None) is None: param.update({'Build_log': False})
    if param.get('Build_sin_cos', None) is None: param.update({'Build_sin_cos': False})
    if param.get('Build_sin_cos_degree', None) is None: param.update({'Build_sin_cos_degree': 3})
    if param.get('Build_sinh_cosh', None) is None: param.update({'Build_sinh_cosh': False})
    if param.get('Build_rbf_gaussian', None) is None: param.update({'Build_rbf_gaussian': False})
    if param.get('Build_all', None) is None: param.update({'Build_all': False})
    if param['Build_all']: param.update({'Build_poly': True, 'Build_root': True, 'Build_xy': True, 'Build_inverse': True, 'Build_log': True, 'Build_sin_cos': True, 'Build_sinh_cosh': True, 'Build_rbf_gaussian': True})

    if tX.size == 0:
        return tX

    tX = tX.reshape(tX.shape[0], -1).copy()
    if np.all(tX[:, 0] == np.ones(tX.shape[0])) == True: tX = np.delete(tX, 0, axis=1)

    tX_base = tX
    n_params = tX_base.shape[1]

    if param['Build_poly']:
        for p in range(n_params):
            new_part = np.power(tX_base[:, p].reshape(-1, 1), np.arange(2, param['Build_poly_degree'] + 1).reshape(1, -1))
            tX = np.hstack((tX, new_part))
    if param['Build_root']:
        for p in range(n_params):
            new_part = tX_base[:, p].reshape(-1, 1)
            new_part[new_part < 0.0] = 0.0
            new_part = np.power(new_part, 1 / np.arange(2, param['Build_root_degree'] + 1).reshape(1, -1))
            tX = np.hstack((tX, new_part))
    if param['Build_xy']:
        for p1 in range(n_params):
            for p2 in range(p1 + 1, n_params):
                new_part = (tX_base[:, p1] * tX_base[:, p2]).reshape(-1, 1)
                tX = np.hstack((tX, new_part))
    if param['Build_inverse']:
        for p in range(n_params):
            new_part = (1.0 / tX_base[:, p]).reshape(-1, 1)
            tX = np.hstack((tX, new_part))
    if param['Build_log']:
        for p in range(n_params):
            new_part = tX_base[:, p].reshape(-1, 1)
            new_part[new_part<=0.0] = 1.0
            new_part = np.log(new_part)
            tX = np.hstack((tX, new_part))
    if param['Build_sin_cos']:
        for p in range(n_params):
            new_part_sin = np.power(np.sin(tX_base[:, p].reshape(-1, 1)), np.arange(1, param['Build_sin_cos_degree'] + 1).reshape(1, -1))
            new_part_cos = np.power(np.cos(tX_base[:, p].reshape(-1, 1)), np.arange(1, param['Build_sin_cos_degree'] + 1).reshape(1, -1))
            tX = np.hstack((tX, new_part_sin, new_part_cos))
    if param['Build_sinh_cosh']:
        for p in range(n_params):
            new_part = tX_base[:, p].reshape(-1, 1)
            tX = np.hstack((tX, np.sinh(new_part), np.cosh(new_part)))
    if param['Build_rbf_gaussian']:
        for p in range(n_params):
            mean = tX_base[:, p].mean()
            std = tX_base[:, p].std()
            new_part = np.exp(-0.5*(((tX_base[:, p]-mean)/std)**2)).reshape(-1,1)
            tX = np.hstack((tX, new_part))

    return np.hstack((np.ones(tX.shape[0]).reshape(-1,1), tX.reshape(tX.shape[0],-1)))

def preprocess_data(y_train, tX_train, ids_train, tX_test, ids_test, param=None):
    '''
    Preprocessing of the data.
    :param y_train: training labels [n_samples_train]
    :param tX_train: training data [n_samples_train x n_dim]
    :param ids_train: training ids of samples [n_samples_train]
    :param tX_test: testing data [n_samples_test x n_dim]
    :param ids_test: testing ids of samples [n_samples_test]
    :param param: dict of different parameters to preprocess the data [dict]
                  default: {'Print_info': False,                     | print informations about the data tX
                            'Remove_missing': False,                 | remove the samples with -999 values (missing values)
                            'Remove_missing_number': -1,             | [-1]: remove the row if there is at least one missing value | [int or list of int]: remove the row(s) with the specified number(s) of missing values
                            'Remove_uniform_parameters': True,       | remove the useless parameters (uniform random parameters)
                            'Remove_zero_variance_parameters': True, | remove the useless parameters (zero variance parameters)
                            'Standardization': True,                 | standardize the data
                            'Normalization_min_max': False,          | normalize the data with min and max -> data values will be between 0.0 and 1.0
                            'Remove_outliers': True,                 | change the values of outliers (no deleting)
                            'Remove_outliers_std_limit': 6.0,        | condition to consider a value as outlier: z-score > 6.0 from the mean (number of std difference from the mean allowed)
                            'Missing_to_0': False,                   | change missing values (-999) and outliers values (if 'Remove_outliers'=True) to 0.0 (0.0 is also the mean if 'Standardization'=True)
                            'Missing_to_median': True,               | change missing values (-999) and outliers values (if 'Remove_outliers'=True) to the median of their parameter
                            'Feature_engineering': True,             | feature engineering of the data -> see associated function feature_engineering(tX, param=None)
                            'Standardization_after_fe': False,       | standardize the data after feature engineering
                            'Normalization_min_max_after_fe': False} | normalize the data with min and max after feature engineering
                  Note: param also contains parameters for feature_engineering. See the associated function for more details.
    :return: data preprocessed (y_train, tX_train, ids_train, tX_test, ids_test)
    '''

    if param is None: param = {}
    if param.get('Print_info', None) is None: param.update({'Print_info': False})  # print informations about the data tX
    if param.get('Remove_missing', None) is None: param.update({'Remove_missing': False})  # remove the samples with -999 values
    if param.get('Remove_missing_number', None) is None: param.update({'Remove_missing_number': -1})  # [-1]: remove the row if there is at least one missing value | [int or list of int]: remove the row(s) with the specified number(s) of missing values
    if param.get('Remove_uniform_parameters', None) is None: param.update({'Remove_uniform_parameters': True})  # remove the useless parameters (uniform random parameters)
    if param.get('Remove_zero_variance_parameters', None) is None: param.update({'Remove_zero_variance_parameters': True})  # remove the useless parameters (zero variance parameters)
    if param.get('Standardization', None) is None: param.update({'Standardization': True})  # standardize the data
    if param.get('Normalization_min_max', None) is None: param.update({'Normalization_min_max': False})  # normalize the data with min and max -> data values will be between 0.0 and 1.0
    if param.get('Remove_outliers', None) is None: param.update({'Remove_outliers': True})  # change the values of outliers (no deleting)
    if param.get('Remove_outliers_std_limit', None) is None: param.update({'Remove_outliers_std_limit': 6.0})  # condition to consider a value as outlier: z-score > 6.0 from the mean (number of std difference from the mean allowed)
    if param.get('Missing_to_0', None) is None: param.update({'Missing_to_0': False})  # change missing values (-999) and outliers values (if 'Remove_outliers'=True) to 0.0 (0.0 is also the mean if 'Standardization'=True)
    if param.get('Missing_to_median', None) is None: param.update({'Missing_to_median': True})  # change missing values (-999) and outliers values (if 'Remove_outliers'=True) to the median of their parameter
    if param['Missing_to_0'] == False and param['Missing_to_median'] == False: param['Missing_to_median'] = True  # one of these two parameters has to be True
    if param.get('Feature_engineering', None) is None: param.update({'Feature_engineering': True})  # feature engineering of the data -> see associated function feature_engineering(tX, param=None)
    if param.get('Standardization_after_fe', None) is None: param.update({'Standardization_after_fe': False})  # standardize the data after feature engineering
    if param.get('Normalization_min_max_after_fe', None) is None: param.update({'Normalization_min_max_after_fe': False})  # normalize the data with min and max after feature engineering

    if tX_train.ndim == 1:
        tX_train = tX_train.reshape((-1, 1))
    if tX_test.ndim == 1:
        tX_test = tX_train.reshape((-1, 1))

    tX = np.vstack((tX_train, tX_test))
    mat_missing = np.full(tX.shape, False)  # matrix [n_samples x n_dim] containing True when tX==-999 and False when tX!=-999
    mat_missing[np.where(tX == -999)] = True

    if param['Print_info']:
        print(f"Minimum value of X_train: {tX_train.min()}\nMaximum value of X_train: {tX_train.max()}")
        values, counts = np.unique(tX_train, return_counts=True)
        print(f"Number of -999.0 (missing values) in X_train: {dict(zip(values, counts)).get(-999.0,0)}")
        values, counts = np.unique((tX_train == -999.0).sum(axis=1), return_counts=True)
        print(f"Number of samples without missing values in X_train: {dict(zip(values, counts)).get(0,0)}/{tX_train.shape[0]}")
        for i in range(values.size):
            print(f"Number of samples with {values[i]} missing values: {counts[i]}")
    if param['Remove_missing']:
        if param['Remove_missing_number'] == -1:
            n_to_remove = np.arange(1,tX.shape[0]+1)
        else:
            n_to_remove = np.array(param['Remove_missing_number'])
        id_good_train = np.invert(np.any(np.count_nonzero(tX_train == -999.0, axis=1).reshape(-1, 1) == n_to_remove.reshape(1, -1), axis=1))
        tX = np.vstack((tX[:tX_train.shape[0],:][id_good_train],tX[-tX_test.shape[0]:,:]))
        y_train = y_train[id_good_train]
        ids_train = ids_train[id_good_train]
        mat_missing = np.full(tX.shape, False)
        mat_missing[np.where(tX == -999)] = True
    if param['Remove_uniform_parameters']:
        param_to_remove = []
        for p in range(tX.shape[1]):
            val = np.delete(tX[:,p], tX[:,p]==-999)
            if val.size != 0:
                val_range = np.linspace(val.min(), val.max(), 101)
                val_density = np.count_nonzero(np.logical_and(val.reshape(-1,1)>val_range[0:100].reshape(1,-1), val.reshape(-1,1)<val_range[1:101].reshape(1,-1)), axis=0)
                if np.abs(val_density-val_density.mean()).max() < 0.25*val_density.mean():
                    param_to_remove.append(p)
        tX = np.delete(tX, param_to_remove, axis=1)
        mat_missing = np.delete(mat_missing, param_to_remove, axis=1)
    if param['Remove_zero_variance_parameters']:
        tX2 = tX.copy()
        tX2[mat_missing] = np.nan
        tX_std = np.nanstd(tX2, axis=0)
        tX = np.delete(tX, np.logical_or(tX_std == 0.0, np.invert(np.isfinite(tX_std))), axis=1)
        mat_missing = np.delete(mat_missing, np.logical_or(tX_std == 0.0, np.invert(np.isfinite(tX_std))), axis=1)
    if param['Standardization']:
        tX2 = tX.copy()
        tX2[mat_missing] = np.nan
        tX_mean = np.nanmean(tX2, axis=0).reshape(1,-1)
        tX_std = np.nanstd(tX2, axis=0).reshape(1,-1)
        tX_std[tX_std == 0.0] = 1.0
        tX = (tX - tX_mean) / tX_std
    if param['Normalization_min_max']:
        min = tX.min(axis=0).reshape(1,-1)
        max = tX.max(axis=0).reshape(1,-1)
        tX = (tX-min)/(max-min)
    if param['Remove_outliers']:
        tX2 = tX.copy()
        tX2[mat_missing] = np.nan
        tX_mean = np.nanmean(tX2, axis=0).reshape(1,-1)
        tX_std = np.nanstd(tX2, axis=0).reshape(1,-1)
        n_std = param['Remove_outliers_std_limit']
        mat_missing[tX < tX_mean - n_std * tX_std] = True
        mat_missing[tX > tX_mean + n_std * tX_std] = True
    if param['Missing_to_0']:
        # Missing_to_0 is also synonymous to Missing_to_mean if standardization is True
        tX[mat_missing] = 0.0
    if param['Missing_to_median']:
        tX2 = tX.copy()
        tX2[mat_missing] = np.nan
        tX_median = np.nanmedian(tX2, axis=0)  # medians of the parameters [n_dim]
        ind_missing = np.where(mat_missing == True)
        tX[ind_missing] = tX_median[ind_missing[1]]
    if param['Feature_engineering']:
        # feature_engineering uses all the other parameters from param variable to do feature expansion
        tX = feature_engineering(tX,param)
    if param['Standardization_after_fe']:
        tX2 = tX[(mat_missing.sum(axis=1) == 0)]
        if tX2.size != 0:
            tX_mean = np.mean(tX2, axis=0).reshape(1,-1)
            tX_std = np.std(tX2, axis=0).reshape(1,-1)
        else:
            tX_mean = np.mean(tX, axis=0).reshape(1,-1)
            tX_std = np.std(tX, axis=0).reshape(1,-1)
        tX_std[tX_std == 0.0] = 1.0
        tX = (tX - tX_mean) / tX_std
    if param['Normalization_min_max_after_fe']:
        min = tX.min(axis=0).reshape(1,-1)
        max = tX.max(axis=0).reshape(1,-1)
        tX = (tX-min)/(max-min)
    tX_train = tX[:-tX_test.shape[0], :]
    tX_test = tX[-tX_test.shape[0]:,:]
    return y_train, tX_train, ids_train, tX_test, ids_test

""""""""""""""""""""""
" LOSS/GRAD/ACCURACY "
"""""""""""""""""""""

def compute_loss_MSE(y, tX, w):
    '''
    Computes the loss for mean squared error (MSE)
    :param y: labels [n_samples]
    :param tX: data [n_samples x n_dim]
    :param w: weights [n_dim]
    :return: mean squared error loss [float]
    '''
    e = y.reshape(-1,1) - tX@(w.reshape(-1, 1))
    loss_MSE = (e.T@e).item()/(2*y.size)
    return loss_MSE

def compute_gradient_MSE(y, tX, w):
    '''
    Computes gradient for mean squared error (MSE)
    :param y: labels [n_samples]
    :param tX: data [n_samples x n_dim]
    :param w: weights [n_dim]
    :return: mean squared error gradient
    '''
    e = y.reshape(-1,1) - tX@(w.reshape(-1, 1)) #dim = n
    gradient = -1/y.size * np.dot(tX.T, e)      #dim = d
    return gradient

def sigmoid(t):
    '''
    Computes the sigmoid function of t: 1/(1+e^(-t))
    :param t: exponent
    :return: sigmoid(t)
    '''
    sig = np.empty(t.shape, dtype=np.float64)  # initialization of sigmoid matrix
    # handling of numerical errors:
    t_modified = t.copy()
    t_modified[np.logical_or(t <= -100,t >= 100)] = 0.0  # t_modified = t exponent modified in order to avoid RuntimeWarning Overflow
    sig[np.logical_and(t < 100, t > -100)] = (1.0 / (1.0 + np.exp(-t_modified)))[np.logical_and(t < 100, t > -100)]  # 1/(1+e^(-t))
    sig[t >= 100] = 1.0  # fix numerical errors: if t>=100, then e^(-t) < 10^43, then we assume 1/(1+e^(-t))=1
    sig[t <= -100] = 0.0  # fix numerical errors: if t<=-100, then e^(-t) > 10^43, then we assume 1/(1+e^(-t))=0
    return sig

def compute_loss_NLL(y, tX, w, lambda_=0.0):
    '''
    Computes the loss for negative log-likelihood (NLL) using labels y in {-1,1}
        Warning: The formula is derived from the course lectures but it's not the same as for labels y in {0,1} !
        Warning: The weights used are therefore specific to the case with y in {-1,1} !
    :param y: labels [n_samples]
    :param tX: data [n_samples x n_dim]
    :param w: weights [n_dim]
    :param lambda_: regularization parameter [float]
    :return: negative log-likelihood loss [float]
    '''
    y = y.reshape(-1, 1)
    tX = tX.reshape(tX.shape[0], -1)
    w = w.reshape(-1, 1)
    t = -tX@w  # exponent
    # handling of numerical errors:
    t_modified = t.copy()
    t_modified[np.logical_or(t<=-100,t>=100)] = 0.0  # t_modified = t exponent modified in order to avoid RuntimeWarning Overflow
    log_part = np.empty(t.shape, dtype=np.float64)  # initialization
    log_part[np.logical_and(t>-100,t<100)] = np.log(1 + np.exp(t_modified))[np.logical_and(t>-100,t<100)] # log(1+e^t)
    log_part[t <= -100] = np.zeros(shape=t.shape)[t <= -100]  # fix numerical errors: if t<=-100, then e^t < 10^43, then we assume log(1+e^t)=0.0
    log_part[t >= 100] = t[t >= 100] # fix numerical errors: if t>=100, then e^t > 10^43, then we assume log(1+e^t)=t
    loss = -0.5*(1.0-y.T)@t + log_part.sum() + lambda_*(w**2).sum()
    # if np.all(loss==np.inf) == True: print("The loss is infinity !")  # debugging line
    return loss

def compute_gradient_NLL(y, tX, w, lambda_=0.0):
    '''
    Computes the gradient of negative log-likelihood (NLL) loss using labels y in {-1,1}
        Warning: The formula is derived from the course lectures but it's not the same as for labels y in {0,1} !
        Warning: The weights used are therefore specific to the case with y in {-1,1} !
    :param y: labels [n_samples]
    :param tX: data [n_samples x n_dim]
    :param w: weights [n_dim]
    :param lambda_: regularization parameter [float]
    :return: negative log-likelihood loss gradient [n_dim]
    '''
    y = y.reshape(-1, 1)
    tX = tX.reshape(tX.shape[0], -1)
    w = w.reshape(-1, 1)
    grad = tX.T @ (0.5*(1.0-y) - sigmoid(-tX @ w)) + 2*lambda_*w
    return grad

def compute_hessian_NLL(y, tX, w, lambda_=0.0):
    '''
    Computes the hessian matrix of negative log-likelihood (NLL) loss using labels y in {-1,1}
    :param y: labels [n_samples]
    :param tX: data [n_samples x n_dim]
    :param w: weights [n_dim]
    :param lambda_: regularization parameter [float]
    :return: negative log-likelihood loss hessian matrix [n_dim x n_dim]
    '''
    y = y.reshape(-1, 1)
    tX = tX.reshape(tX.shape[0], -1)
    w = w.reshape(-1, 1)
    S = (sigmoid(-tX @ w) * (1.0 - sigmoid(-tX @ w))).reshape(1,-1)
    H = (tX.T*S)@tX + np.diag(np.full((w.shape[0],),2*lambda_))
    return H

def compute_accuracy(y, tX, w):
    '''
    Computes accuracy.
    :param y: true labels [n_samples]
    :param tX: data [n_samples x n_dim]
    :param w: weights [n_dim]
    :return: accuracy [float]
    '''
    if y.size == 0 or tX.size == 0: return 0.0
    y = y.reshape(y.shape[0], -1)
    tX = tX.reshape(tX.shape[0], -1)
    w = w.reshape(w.shape[0], -1)
    accuracy = np.mean(y == predict_labels(w, tX))
    return accuracy

"""""""""""""""""""""""""""""""""""""""""
" GRADIENT DESCENT ADDITIONAL FUNCTIONS "
"""""""""""""""""""""""""""""""""""""""""

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    '''
    Generate a minibatch iterator for a dataset (used in GD/SGD).
    :param y: labels
    :param tx: data
    :param batch_size: batch_size [int]
    :param num_batches: total number of batches [int]
    :param shuffle: shuffling of the data [bool]
    :return: minibatch iterator
    '''
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

def ada_grad(gradient, h, gamma_zero):
    '''
    AdaGrad implementation for the learning rate (used in gradient descent)
    :param gradient: gradient
    :param h: parameter containing actual state of AdaGrad (h[t] is computed using h[t-1] and gradient)
    :param gamma_zero: initial gamma (constant)
    :return: gamma (learning rate), h (should be used for the next iteration of AdaGrad)
    '''
    gradient = gradient.reshape(-1,1)
    h = h.reshape(-1,1)
    gamma_zero = gamma_zero.reshape(-1,1)
    h+=np.power(gradient, 2)
    gamma=gamma_zero*(1/np.sqrt(h))
    return gamma, h

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

def cross_validation(y, x, k_indices, k, model, degree=1, params=None, params_logistic=None, feedback=False):
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
    """
    params_logistic.update({'AdaGrad': True})
    # Recap of the arguments entered as the function is heavy in parameters
    if feedback:
        print('Starting cross-validation {}/{} for {}, extended feature of degree {} and arguments : {}'.format(k + 1,len(k_indices),model,degree,params))

    # Create k-th split of train/test sets, possibly with extended features
    train_folds = list(range(k_indices.shape[0]))
    train_folds.remove(k)
    train_idx = np.concatenate(([k_indices[fold, :] for fold in train_folds]))
    test_idx = k_indices[k, :]

    feat_matrix_tr = build_poly(x[train_idx], degree)
    feat_matrix_te = build_poly(x[test_idx], degree)
    y_tr = y[train_idx]
    y_te = y[test_idx]
    initial_w = np.random.normal(0.0, 0.1, size=(feat_matrix_tr.shape[1], 1))

    # Use model given in parameter and initialize relevant parameters
    if model == 'least_squares':
        w, loss_tr = least_squares(y_tr, feat_matrix_tr)
    elif model == 'least_squares_GD':
        max_iters, plot = params['max_iters'], params['plot']
        gamma_zero = 0.1 * np.ones(feat_matrix_tr.shape[1])
        w, loss_tr = least_squares_GD(y_tr, feat_matrix_tr, initial_w, max_iters, gamma_zero, plot, ada_grad=True)
    elif model == 'least_squares_SGD':
        max_iters, batch_size, plot = params['max_iters'], params['batch_size'], params['plot']
        gamma_zero = 0.01 * np.ones(feat_matrix_tr.shape[1])
        w, loss_tr = least_squares_SGD(y_tr, feat_matrix_tr, initial_w, max_iters, gamma_zero, batch_size, plot, ada_grad=True)
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
    params_logistic.update({'AdaGrad': True})
    # split data in k_fold:
    k_indices = build_k_indices(y, k_fold, seed)
    accs_tr = []
    accs_te = []
    losses_tr = []
    losses_te = []

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
                loss_tr, loss_te, acc_tr, acc_te = cross_validation(y, x, k_indices, k, model, degree, params,params_logistic=params_logistic)
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
                    loss_tr, loss_te, acc_tr, acc_te = cross_validation(y, x, k_indices, k, model, degree, params,params_logistic=params_logistic)
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

