import time
import os
import datetime
from Implementations import *

def multi_ridge_regression(y_train_, tX_train_, ids_train_, tX_test_, ids_test_):
    y_train_raw = y_train_.copy()
    tX_train_raw = tX_train_.copy()
    ids_train_raw = ids_train_.copy()
    tX_test_raw = tX_test_.copy()
    ids_test_raw = ids_test_.copy()

    def separations_conditions(separations):
        def combine_conditions(cond_list):
            final_cond = cond_list[0]
            for i in range(1,len(cond_list)):
                final_cond = np.logical_and(final_cond, cond_list[i])
            return final_cond
        separations_cond_train = []
        separations_cond_test = []
        for p, lim in separations.items():
            separations_cond_train.append([tX_train_raw[:, p] <= lim, tX_train_raw[:, p] > lim])
            separations_cond_test.append([tX_test_raw[:, p] <= lim, tX_test_raw[:, p] > lim])
        n_cond = 2**len(separations)
        sep_cond_train = []
        sep_cond_test = []
        for i in range(n_cond):
            path = bin(i)[2:].rjust(len(separations),'0')
            cond_to_combine_train = []
            cond_to_combine_test = []
            for j in range(len(path)):
                cond_to_combine_train.append(separations_cond_train[j][int(path[j])])
                cond_to_combine_test.append(separations_cond_test[j][int(path[j])])
            final_cond_train = combine_conditions(cond_to_combine_train)
            final_cond_test = combine_conditions(cond_to_combine_test)
            sep_cond_train.append(final_cond_train)
            sep_cond_test.append(final_cond_test)
        return sep_cond_train, sep_cond_test

    separations = {0: -0.3, 7: 1.25}
    sep_cond_train, sep_cond_test = separations_conditions(separations)

    index_train = []
    index_test = []
    for PRI_jet_num in np.unique(tX_train_raw[:,22]):
        for i in range(len(sep_cond_train)):
            index_train.append(np.logical_and((tX_train_raw[:, 22] == PRI_jet_num), sep_cond_train[i]))
            index_test.append(np.logical_and((tX_test_raw[:, 22] == PRI_jet_num), sep_cond_test[i]))

    samples_done = 0
    predicted_y = np.array([])
    parameters = {3:{'Remove_outliers_std_limit':15},7:{'Remove_outliers_std_limit':5},11:{'Remove_outliers_std_limit':13},12:{'Remove_outliers_std_limit':4},15:{'Remove_outliers_std_limit':4}}
    for i in range(len(index_train)):
        t1 = time.time()
        parameter = parameters.get(i, {})
        if parameter.get('Remove_outliers_std_limit', None) is None: parameter.update({'Remove_outliers_std_limit': 6.0})
        parameter.update({'Build_all':True,'Build_poly_degree':9})
        y1, tX1, ids1, tX2, ids2 = preprocess_data(y_train_raw[index_train[i]], tX_train_raw[index_train[i]], ids_train_raw[index_train[i]], tX_test_raw[index_test[i]], ids_test_raw[index_test[i]], param=parameter)
        w = np.random.normal(0.0, 0.1, size=(tX1.shape[1], 1))

        if tX1.size != 0:
            w, _ = ridge_regression(y1.reshape(-1), tX1, 0.001)

        y2 = predict_labels(w.reshape(-1, 1), tX2.reshape(tX2.shape[0], -1))
        if predicted_y.size == 0:
            predicted_y = np.hstack((ids2.reshape(-1,1),y2))
        else:
            new_part = np.hstack((ids2.reshape(-1,1),y2))
            predicted_y = np.vstack((predicted_y,new_part))

        samples_done += tX2.shape[0]
        t2 = time.time()
        print(f"{i + 1}/16 subset done: {samples_done}/{tX_test_raw.shape[0]} samples done ({round(t2-t1,2)} sec)")

    predicted_y = predicted_y[np.argsort(predicted_y[:,0])]
    predicted_y = predicted_y[:,1]
    return predicted_y

if __name__ == '__main__':
    np.random.seed(1)

    if os.path.isdir('Datasets'):
        DATA_TRAIN_PATH = 'Datasets/train.csv'
    elif os.path.isdir('data'):
        DATA_TRAIN_PATH = 'data/train.csv'
    elif os.path.isdir('../Projet_1_data'):
        DATA_TRAIN_PATH = '../Projet_1_data/train.csv'
    else:
        print("The Datasets folder cannot be found !")
    print("Loading training data...")
    t1 = time.time()
    y_train, tX_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
    t2 = time.time()
    print(f"Training data loaded ! ({round(t2 - t1, 2)} sec)")

    if os.path.isdir('Datasets'):
        DATA_TEST_PATH = 'Datasets/test.csv'
    elif os.path.isdir('data'):
        DATA_TEST_PATH = 'data/test.csv'
    elif os.path.isdir('../Projet_1_data'):
        DATA_TEST_PATH = '../Projet_1_data/test.csv'
    else:
        print("The Datasets folder cannot be found !")
    print("Loading testing data...")
    t1 = time.time()
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    t2 = time.time()
    print(f"Testing data loaded ! ({round(t2 - t1, 2)} sec)")

    y_pred = multi_ridge_regression(y_train, tX_train, ids_train, tX_test, ids_test)

    timestamp = str(datetime.datetime.now()).replace(' ', '_').replace(':', 'h')[:16]
    OUTPUT_PATH = f'Results/predictions_{timestamp}.csv'
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

