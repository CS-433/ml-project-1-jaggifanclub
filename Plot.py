import math
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_param_vs_err(params, err_tr, err_te, model_name = 'model', err_type = 'MSE', param = 'degree', save_img = False, img_name = '-1'):
    """
    Visualization of the curves of mse/accuracy given parameter (degree, lambda or other).
     :param params: list of the parameters used for each version of the model
     :param err_tr: corresponding training error, whether mse or accuracy
     :param err_te: corresponding test error, whether mse or accuracy
     :param param: label of the parameter used
     :param err_type: type of error (mse or accuracy)
     :param model_name: name of the model used
     :param save_img: boolean indicating if the image generated must be saved
     :param img_name: if the image must be saved, name demanded (in order to not erase previously saved images)
    """
    if err_type == 'MSE' or err_type == 'mse':
        best_idx = np.argmin(err_te)
    elif err_type == 'accuracy' or err_type == 'Accuracy' or err_type == 'ACCURACY':
        best_idx = np.argmax(err_te)
    
    if param == 'lambda':
        plt.semilogx(params, err_tr, marker=".", color='b', label='train set')
        plt.semilogx(params, err_te, marker=".", color='r', label='test set')
    else:
        plt.plot(params, err_tr, marker=".", color='b', label='train set')
        plt.plot(params, err_te, marker=".", color='r', label='test set')
    plt.axvline(params[best_idx], color = 'k', ls = '--', alpha = 0.5, label = 'best ' + param)
    plt.xlabel(param)
    plt.ylabel(err_type)
    plt.title(err_type + ' of ' + model_name + ' given different values for parameter: ' + param)
    plt.legend()
    plt.grid(True)
    if save_img:
        if img_name == '-1':
            print('Argument not found: img_name. Image not saved.')
        else:
            plt.savefig('figures/' + img_name)
    plt.show()

def plot_param_vs_loss_and_acc(params, loss_tr, loss_te, acc_tr, acc_te, model_name = 'model', param = 'degree', save_img = False, img_name = '-1'):
    """
    Visualization of the curves of loss AND accuracy given parameter (degree, learning rate, lambda).
     :param params: list of the parameters used for each version of the model
     :param loss_tr: corresponding training loss
     :param loss_te: corresponding test loss
     :param acc_tr: corresponding training accuracy
     :param acc_te: corresponding test accuracy
     :param param: label of the parameter used
     :param model_name: name of the model used
     :param save_img: boolean indicating if the image generated must be saved
     :param img_name: if the image must be saved, name demanded (in order to not erase previously saved images)
    """
    
    best_idx_loss = np.argmin(loss_te)
    best_idx_acc = np.argmax(acc_te)
    
    fig, axs = plt.subplots(1, 2, figsize = [12,5])
    fig.suptitle('Loss and accuracy of ' + model_name + ' given different values for parameter: ' + param)
    if param == 'lambda':
        axs[0].semilogx(params, loss_tr, marker=".", color='b', label='train set')
        axs[0].semilogx(params, loss_te, marker=".", color='r', label='test set')
        axs[1].semilogx(params, acc_tr, marker=".", color='b')
        axs[1].semilogx(params, acc_te, marker=".", color='r')    
    else:
        axs[0].plot(params, loss_tr, marker=".", color='b', label='train set')
        axs[0].plot(params, loss_te, marker=".", color='r', label='test set')
        axs[1].plot(params, acc_tr, marker=".", color='b')
        axs[1].plot(params, acc_te, marker=".", color='r')
    axs[0].axvline(params[best_idx_loss], color = 'k', ls = '--', alpha = 0.5, label = 'best ' + param)
    axs[1].axvline(params[best_idx_acc], color = 'k', ls = '--', alpha = 0.5)
    
    axs[0].set_xlabel(param)
    axs[1].set_xlabel(param)
    axs[0].set_ylabel('Loss')
    axs[1].set_ylabel('Accuracy')
    axs[0].grid(True)
    axs[1].grid(True)
    fig.legend()
    if save_img:
        if img_name == '-1':
            print('Argument not found: img_name. Image not saved.')
        else:
            fig.savefig('figures/' + img_name)
    plt.show()
    
def plot_boxplots(errors, model_names, err_type = 'Accuracy', save_img = False, img_name = '-1'):
    """
    Visualisation of the performance of models across folds.
     :param errors: array of losses/accuracies, such that each ROW contains the losses/accuracies of a same model on different folds (cross-validation)
     :param model_names: names of the models corresponding to each row
     :param err_type: type of error (loss or accuracy)
     :param save_img: boolean indicating if the image generated must be saved
     :param img_name: if the image must be saved, name demanded (in order to not erase previously saved images)
    """
    errors = errors.T
    plt.figure(figsize=(10,4))
    bp = plt.boxplot(errors, labels = model_names, showmeans = True)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
    plt.title('Boxplot of the ' + err_type + ' models (' + str(np.array(errors).shape[0]) + ' folds)')
    plt.ylabel(err_type)
    #plt.xticks(rotation=90)
    if save_img:
        if img_name == '-1':
            print('Argument not found: img_name. Image not saved.')
        else:
            plt.savefig('figures/' + img_name)
    plt.show()
    
def plot_twice_boxplots(losses, accuracies, model_names, save_img = False, img_name = '-1'):
    """
    Visualisation of the performance of models across folds.
     :param losses: array of losses. Each ROW contains the loss of a same model on different folds (cross-validation)
     :param accuracies: array of accuraciess. Each ROW contains the accuracy of a same model on different folds (cross-validation)
     :param model_names: names of the models corresponding to each row
     :param save_img: boolean indicating if the image generated must be saved
     :param img_name: if the image must be saved, name demanded (in order to not erase previously saved images)
    """
    losses = losses.T
    accuracies = accuracies.T
    fig, axs = plt.subplots(1, 2, figsize = [12,5])
    fig.suptitle('Boxplot of the loss and accuracy of models (' + str(np.array(losses).shape[1]) + ' folds)')
    axs[0].boxplot(losses, labels = model_names, showmeans = True)
    axs[0].set_ylabel('Loss')
    bp = axs[1].boxplot(accuracies, labels = model_names, showmeans = True)
    axs[1].set_ylabel('Accuracy')
    fig.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
    if save_img:
        if img_name == '-1':
            print('Argument not found: img_name. Image not saved.')
        else:
            fig.savefig('figures/' + img_name)
    plt.show()
    
def plot_heatmap(err_tr, err_te, degrees, lambdas, model_name, measure_type = 'Accuracy', save_img = False, img_name = '-1'):
    """
    Visualisation of accuracy/loss computed over all lambda-degrees combinations using a heatmap
    :param err_tr: matrix of losses/accuracies computed on training set
    :param err_te: matrix of losses/accuracies computed on test set
    :param degrees: vector of all degrees used to create feature on data before training
    :param lambdas: vector of all lambdas used to regularize training
    :param model_name: model type used to train on data and predict labels
    :param measure_type: Measure used to assess performance (MSE/NLL/Accuracy)
    :param save_img: boolean indicating if the image generated must be saved
    :param img_name: if the image must be saved, name demanded (in order to not erase previously saved images)
    """
    fig, axs = plt.subplots(1, 2, figsize = [15,8])
    fig.suptitle(measure_type + ' of ' + model_name + ' given different values for parameter lambda and degree.')
    
    for i in range(2):
        axs[i].imshow(err_tr, cmap = 'PiYG')
        axs[i].set_xticks(np.arange(len(lambdas)))
        axs[i].set_yticks(np.arange(len(degrees)))
        axs[i].set_xticklabels(lambdas)
        axs[i].set_yticklabels(degrees)
        axs[i].set_xlabel('\u03BB')
        axs[i].set_ylabel('degree')
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
        
    # Write accuracy values
    for i in range(len(degrees)):
        for j in range(len(lambdas)):
            text = axs[0].text(j, i, round(err_tr[i, j], 3),
                           ha="center", va="center", color="k")
            text = axs[1].text(j, i, round(err_te[i, j], 3),
                           ha="center", va="center", color="k")

    axs[0].set_title("Train " + measure_type)
    axs[1].set_title("Test " + measure_type)
    if save_img:
        if img_name == '-1':
            print('Argument not found: img_name. Image not saved.')
        else:
            fig.savefig('figures/' + img_name)
    plt.show()


def plot_features_visualization(y, tX, y_pred=None, names=None, save=True, stacked_plot=True):
    '''
    Function used to visualize the dataset. Label y=1 is Higgs boson and label y=-1 is Other.
    Different possiblities of plotting:
      1) y_pred not provided                 -> 2 superimposed histograms to visualize Higgs boson and Other distributions
      2) y_pred provided, stacked_plot=True  -> stacked area chart to observe true/false positive/negative distributions
      3) y_pred provided, stacked_plot=False -> 2 superimposed histograms to visualize Higgs boson and Other distributions (correct/wrong predictions included)
    :param y: true labels [n_samples]
    :param tX: data [n_samples x n_dim]
    :param y_pred: (optional) predicted labels [n_samples]
                   If provided, the plots will include the true/false predictions
    :param names: (optional) names of the parameters [n_dim]
                  If provided, the plots will take the names as titles
    :param save: If True, the plot will be saved at location figures/Data_visualization/...
    :param stacked_plot: If True, the plot will be a stacked area chart (see above explanations)
    :return: 0
    '''
    y = y.reshape(-1)
    tX = tX.reshape(tX.shape[0], -1)
    tX_positive = tX[y == 1]
    tX_negative = tX[y == -1]
    if y_pred is not None:
        y_pred = y_pred.reshape(-1)
        tX_true_positive = tX[np.logical_and(y == 1, y_pred == 1)]  # Higgs, good prediction
        tX_false_negative = tX[np.logical_and(y == 1, y_pred == -1)]  # Higgs, bad prediction
        tX_true_negative = tX[np.logical_and(y == -1, y_pred == -1)]  # not Higgs, good prediction
        tX_false_positive = tX[np.logical_and(y == -1, y_pred == 1)]  # not Higgs, bad prediction
    plt.close()
    n_rows = math.ceil(tX.shape[1]/3)
    fig, axs = plt.subplots(n_rows, 3, figsize=[45, n_rows*7.5])
    fig.patch.set_facecolor('#E0E0E0')
    bins = np.linspace(tX.min(axis=0), tX.max(axis=0), 101).T
    for i in range(3*n_rows):
        if n_rows > 1:
            ax_index = (i//3, i%3)
        else:
            ax_index = (i%3)
        if i >= tX.shape[1]:
            axs[ax_index].set_visible(False)
        else:
            axs[ax_index].set_xlabel("parameter value")
            axs[ax_index].set_ylabel("density")
            if y_pred is None:
                set5_positive = np.delete(tX_positive[:, i], np.where(tX_positive[:, i] == 0.0))
                set6_negative = np.delete(tX_negative[:, i], np.where(tX_negative[:, i] == 0.0))
                axs[ax_index].hist([set5_positive, set6_negative], bins=bins[i], alpha=0.8, density=True, color=['tab:blue','tab:orange'], stacked=False)
                axs[ax_index].legend(['Higgs boson', 'Other'])
            else:
                set1_higgs_good = np.delete(tX_true_positive[:, i], np.where(tX_true_positive[:, i] == 0.0))
                set2_higgs_bad = np.delete(tX_false_negative[:, i], np.where(tX_false_negative[:, i] == 0.0))
                set3_not_good = np.delete(tX_true_negative[:, i], np.where(tX_true_negative[:, i] == 0.0))
                set4_not_bad = np.delete(tX_false_positive[:, i], np.where(tX_false_positive[:, i] == 0.0))
                if stacked_plot:
                    axs[ax_index].hist([set2_higgs_bad, set4_not_bad, set3_not_good, set1_higgs_good], bins=bins[i], alpha=0.8, density=True, color=['red', 'salmon', 'tab:green', 'darkgreen'], stacked=True)
                    axs[ax_index].legend(['Higgs boson wrongly predicted (false negative)', 'Other wrongly predicted (false positive)', 'Other correctly predicted (true negative)', 'Higgs boson correctly predicted (true positive)'])
                else:
                    axs[ax_index].hist([set2_higgs_bad, set1_higgs_good], bins=bins[i], alpha=0.8, density=True, align='left', rwidth=0.4, color=['darkblue','tab:blue'], stacked=True)
                    axs[ax_index].hist([set4_not_bad, set3_not_good], bins=bins[i], alpha=0.8, density=True, align='mid', rwidth=0.4, color=['darkorange','peachpuff'], stacked=True)
                    axs[ax_index].legend(['Higgs boson wrongly predicted (false negative)','Higgs boson correctly predicted (true positive)','Other wrongly predicted (false positive)', 'Other correctly predicted (true negative)'])
            if names is None:
                axs[ax_index].set_title(f"Parameter {i+1}")
            else:
                axs[ax_index].set_title(f"{i+1}) {names[i]}")
    plt.show()
    if save:
        if os.path.isdir("figures") == False: os.makedirs("figures")
        if os.path.isdir("figures/Data_visualization") == False: os.makedirs("figures/Data_visualization")
        filename_already_used = True
        index = 0
        while filename_already_used:
            index += 1
            filename = f"figures/Data_visualization/data_visualization_{str(index).rjust(3,'0')}.png"
            if os.path.isfile(filename) == False:
                fig.savefig(filename, dpi=300)
                filename_already_used = False
    return 0

