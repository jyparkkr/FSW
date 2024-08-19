from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from .utils_optimization import optCE, optSAGD, optSCIPY, optSCIPY_bivar, optSCIPY_bivar_bis
from scipy.special import softmax
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score # weighted
import matplotlib.pyplot as plt
# from statsmodels.distributions.empirical_distribution import ECDF
from .synthetic_data import make_unfair_poolclassif, data_viz_tsne
import time
import seaborn as sns
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from .multiclass_performance import plot_confusion_matrix

#------------------------------------------------------------#
# utils for multi-class fairness
#------------------------------------------------------------#

# def get_multiclass_performance(model, X_train, X_test, y_train, y_test, print_results = True):

#     # define the ovr strategy
#     ovr = OneVsRestClassifier(model)

#     # fit model
#     ovr.fit(X_train[:, :-1], y_train)

#     # make predictions
#     yhat_proba = ovr.predict_proba(X_test[:, :-1]) # not a softmax, just a normalization on the positives probs after OvR
#     yhat = ovr.predict(X_test[:, :-1])

#     print("test :", accuracy_score(y_test, ovr.predict(X_test[:, :-1])))
#     print("train :", accuracy_score(y_train, ovr.predict(X_train[:, :-1])))

#     C = confusion_matrix(y_test, yhat)
#     plot_confusion_matrix(C)
    
#     return score_test, score_train

def ECDF(*args, **kwargs):
    raise NotImplementedError

def print_fairness_results(acc_dict, ks_dict, time_dict = None, model_name = ""):
    """
    print fairness results
    """
    print("Accuracies")
    for name in acc_dict.keys():
        print(f'{model_name} {name} : {round(np.mean(acc_dict[name]), 2)} +-{round(np.std(acc_dict[name]), 3)}')
    print()
    print("Unfairness")
    for name in ks_dict.keys():
        print(f'{model_name} {name} : {round(np.mean(ks_dict[name]), 2)} +-{round(np.std(ks_dict[name]), 2)}')
    if time_dict is not None:
        print()
        print("Times")
        for name in time_dict.keys():
            print(f'{model_name} {name} : {round(np.mean(time_dict[name]), 3)} +-{round(np.std(time_dict[name]), 3)}')
    return None

def prepare_fairness(X_pool, sen_pool, prob_pool):
    X0 = X_pool[sen_pool==0]
    X1 = X_pool[sen_pool==1]
    ps = np.array([len(X0), len(X1)])/len(X_pool)
    y_prob_dict = dict()
    y_prob_dict[0] = prob_pool[sen_pool==0]
    y_prob_dict[1] = prob_pool[sen_pool==1]
    return y_prob_dict, ps


def unfairness(data1, data2):
    """
    compute the unfairness of two populations
    """
    K = int(np.max((np.max(data1), np.max(data2))))+1
    nu_0 = np.zeros(K)
    nu_1 = np.zeros(K)

    pos, counts = np.unique(data1, return_counts=True)
    nu_0[pos.astype(int)] = counts/len(data1)

    pos, counts = np.unique(data2, return_counts=True)
    nu_1[pos.astype(int)] = counts/len(data2)
    
    #unfair_value = np.abs(nu_0 - nu_1).sum()
    unfair_value = np.abs(nu_0 - nu_1).max()
    return unfair_value

    

#------------------------------------------------------------#
# two Fairness methods
#------------------------------------------------------------#
    
def fair_soft_max(X_test, X_pool, ovr, c = 0.1, opt = "SAGD", sigma = 10**(-5), epsilon_fair = 0.1, \
                  print_lambda=False, prob_pool=None, sen_pool=None, prob=None, sen=None, n_classes=None):
    """
    for the optimization technique we have "CE" (cross-entropy) or "SAGD" (smoothed accelerated GD)
    ovr: classifier
    """
    start_time = time.time()

    # computation of lambda (soft and hard)
    y_prob_dict, ps = prepare_fairness(X_pool, sen_pool, prob_pool)
    # try:    
    #     n_classes = ovr.n_classes_
    # except:
    #     n_classes = len(ovr.classes_)

    def lam_fairness_soft(lam, c = c):
        res = 0
        for s in [0,1]:
            val = y_prob_dict[s]*ps[s] - (2*s-1)*lam
            res += np.mean(np.sum(softmax(val/c, axis=1)*val, axis=1)) # Smooth arg max
        return(res)

    def bivar_fairness_soft(lam, n_classes = n_classes, c = c):
        res = 0
        lamb = lam[:n_classes]
        beta = lam[n_classes:]
        for s in [0,1]:
            val = y_prob_dict[s]*ps[s] - (2*s-1)*(lamb-beta)
            res += np.mean(np.sum(softmax(val/c, axis=1)*val, axis=1)) # Smooth arg max
        res += epsilon_fair * np.sum(lamb+beta)
        return(res)

    def nablaG(lam, c = c):
        res = 0
        for s in [0,1]:
            val = y_prob_dict[s]*ps[s] - (2*s-1)*lam
            softmax_val = softmax(val/c, axis=1)
            res -= (2*s-1) * np.mean( softmax_val, axis = 0) # Smooth arg max
        return(res)

    #def nablaGlam(lam, beta, epsilon = epsilon_fair, c = c):
    #    res = 0
    #    for s in [0,1]:
    #        val = y_prob_dict[s]*ps[s] - (2*s-1)*(lam-beta)
    #        softmax_val = softmax(val/c, axis=1)
    #        res -= (2*s-1) * np.mean(np.sum(softmax_val*val, axis = 0))  # Smooth arg max
    #    res += epsilon#*np.sum(lam)
    #    return(res)
    #def nablaGbeta(lam, beta, epsilon = epsilon_fair, c = c):
    #    res = 0
    #    for s in [0,1]:
    #        val = y_prob_dict[s]*ps[s] - (2*s-1)*(lam-beta)
    #        softmax_val = softmax(val/c, axis=1)
    #        res += (2*s-1) * np.mean( softmax_val, axis = 0) # Smooth arg max
    #    res += epsilon#*np.sum(beta)
    #    return(res)
    
    if opt == "CE":
        lam_soft = optCE(fun = lam_fairness_soft, n = 2000, d = n_classes, eps = 0.001, max_iter = 100, print_results = False)
        beta_soft = np.zeros(len(lam_soft))
    elif opt == "SAGD":
        lam_soft = optSAGD(nablaG, n_classes, c = c, T = 1000)
        beta_soft = np.zeros(lam_soft.shape)
    #elif opt == "SAGD_bivar":
    #    lam_soft, beta_soft = optSAGD_bivar(nablaGlam, nablaGbeta, n_classes, epsilon = epsilon_fair, c = c, T = 1000)
    #    #print(lam_soft)
    #    #print(beta_soft)
    elif opt == "optim":
        lam_soft = optSCIPY(fun = lam_fairness_soft, n_classes = n_classes)
        beta_soft = np.zeros(len(lam_soft))
    elif opt == "optim_bivar":
        lam_soft, beta_soft = optSCIPY_bivar(fun = bivar_fairness_soft, n_classes = n_classes)
    elif opt == "optim_bivar_bis":
        lam_soft, beta_soft = optSCIPY_bivar_bis(fun = bivar_fairness_soft, n_classes = n_classes)
    # inference with and without fairness
    # index_0 = np.where(X_test[:,-1] == -1)[0]
    # index_1 = np.where(X_test[:,-1] == 1)[0]
    index_0 = np.where(sen == 0)[0]
    index_1 = np.where(sen == 1)[0]

    # y_probs = ovr.predict_proba(X_test[:,:-1])
    # y_preds = ovr.predict(X_test[:,:-1])
    y_probs = prob
    y_preds = None
    
    eps = np.random.uniform(0, sigma, (y_probs.shape))
    y_prob_fair_soft = np.zeros(y_probs.shape)
    y_prob_fair_soft[index_0] 
    y_prob_fair_soft[index_0] = ps[0]*(y_probs[index_0]+eps[index_0]) - (-1)*(lam_soft-beta_soft)
    y_prob_fair_soft[index_1] = ps[1]*(y_probs[index_1]+eps[index_1]) - 1*(lam_soft-beta_soft)
    y_pred_fair_soft = np.argmax(y_prob_fair_soft, axis = 1)

    if print_lambda:
        print("lamb:", lam_soft.round(5))
        print("beta:", beta_soft.round(5))

    # track the time
    time_soft = time.time() - start_time

    return y_pred_fair_soft, y_prob_fair_soft, index_0, index_1, y_preds, y_probs, time_soft

#------------------------------------------------------------#
# experimentations and analysis of multi-class fairness
#------------------------------------------------------------#

"""
2.2.1. Method 1 : trust-constr

for epsilon_fair in np.arange(0, 0.18, 0.02):
    print("---------------------------")
    print("epsilon = ", round(epsilon_fair, 2))
    print("---------------------------")
    accs, kss, times, ind0, ind1, yd, yb, ydfh, ydfs, ybfh, ybfs = run_fairness_experimentation(
        RandomForestClassifier(),
        X,
        y,
        X_pool,
        n_times = 1,
        print_results = False,
        c = 0.005,
        soft_opt = "optim_bivar",
        do_ovr = False,
        epsilon_fair = epsilon_fair)

2.2.2. Method 2 : SLSQP
epsilon_interval = [0, 0.1, 0.15]

for epsilon_fair in epsilon_interval:
    print("---------------------------")
    print("epsilon = ", round(epsilon_fair, 2))
    print("---------------------------")
    accs, kss, times, ind0, ind1, yd, yb, ydfh, ydfs, ybfh, ybfs = run_fairness_experimentation(
        RandomForestClassifier(),
        X,
        y,
        X_pool,
        n_times = 30, #n_times=30
        print_results = False,
        c = 0.005,
        soft_opt = "optim_bivar_bis",
        do_ovr = False,
        epsilon_fair = epsilon_fair)

"""

def run_fairness_experimentation(clf, X, X_pool=None, n_times = 1, print_results = True, c = 0.01, 
                                 soft_opt = "optim_bivar_bis", compute_hard = False, do_ovr = True, 
                                 sigma = 10**(-5), epsilon_fair = 0, compute_baseline=False, 
                                 X_train_full=None, y_train_full=None, print_lambda=False,
                                 Xtrain=None, prob_train=None, sen_train=None, prob=None, sen=None, n_classes=None):
    """
    fairness under misclassification risk (computation of lambda)
    opt in ["CE", "SADG", "SAGD_bivar"]
    """
    # ks_dict = dict()
    # ks_dict["unfair"]    = []
    # ks_dict["fair_soft"] = []

    # acc_dict = dict()
    # acc_dict["unfair"]    = []
    # acc_dict["fair_soft"] = []

    # time_dict = dict()
    # time_dict["fair_soft"] = []
    # if compute_hard:
    #     ks_dict["fair_hard"] = []
    #     acc_dict["fair_hard"] = []
    #     time_dict["fair_hard"] = []

    if n_classes is None:
        n_classes = 10
        print(f"{n_classes=}")
    # if n_classes == 2 and compute_baseline:
    #     ks_dict["fair_baseline"] = []
    #     acc_dict["fair_baseline"] = []
    #     time_dict["fair_baseline"] = []

    X_test = X

    # sensitive attribute should be removed from X_train
    X_train = Xtrain
    for i in range(n_times):
        # if i%10==0 and print_results:
        #     print("ite :", i)

        # train-test-split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # define the ovr strategy and fit the model
        start_time = time.time()
        # if do_ovr:
        if False:
            clf_multi = OneVsRestClassifier(clf).fit(X_train[:, :-1], y_train)
        elif clf is None:
            clf_multi = None
        else:
            raise NotImplementedError
            clf_multi = clf.fit(X_train[:, :-1], y_train)

        training_time = time.time() - start_time
        # inference with and without fairness
        # if X_pool is not None:
        if False:
            if i != 0:
                print_lambda=False
            y_pred_fair_soft, y_prob_fair_soft, index_0, index_1, y_preds, y_probs, time_soft =\
                  fair_soft_max(X_test, X_pool, clf_multi, c = c, opt = soft_opt, sigma=sigma, epsilon_fair=epsilon_fair, print_lambda=print_lambda)
        else:
            if i != 0:
                print_lambda=False
            y_pred_fair_soft, y_prob_fair_soft, index_0, index_1, y_preds, y_probs, time_soft =\
                  fair_soft_max(X_test, X_train, clf_multi, c = c, opt = soft_opt, sigma=sigma, \
                                epsilon_fair=epsilon_fair, print_lambda=print_lambda, \
                                prob_pool=prob_train, sen_pool=sen_train, prob=prob, sen=sen, n_classes=n_classes)

        return y_pred_fair_soft
        
    #     if compute_hard:
    #         if X_pool is not None:
    #             y_pred_fair_hard, y_prob_fair_hard, index_0, index_1, y_preds, y_probs, time_hard =\
    #                   fair_hard_max(X_test, X_pool, clf_multi, sigma=sigma)
    #         else:
    #             y_pred_fair_hard, y_prob_fair_hard, index_0, index_1, y_preds, y_probs, time_hard =\
    #                   fair_hard_max(X_test, X_train[:, :-1], clf_multi, sigma=sigma)
    #         ks_dict["fair_hard"].append( unfairness(y_pred_fair_hard[index_0], y_pred_fair_hard[index_1]) )
    #         acc_dict["fair_hard"].append( accuracy_score(y_test, y_pred_fair_hard) )
    #         time_dict["fair_hard"].append(time_hard+training_time)
    #     else:
    #         y_pred_fair_hard = None, 
    #         y_prob_fair_hard = None

    #     # keep the results
    #     ks_dict["unfair"].append( unfairness(y_preds[index_0], y_preds[index_1]) )
    #     ks_dict["fair_soft"].append( unfairness(y_pred_fair_soft[index_0], y_pred_fair_soft[index_1]) )
    #     acc_dict["unfair"].append( accuracy_score(y_test, y_preds) )
    #     acc_dict["fair_soft"].append( accuracy_score(y_test, y_pred_fair_soft) )
    #     time_dict["fair_soft"].append(time_soft+training_time)
        
    #     # if n_classes==2 and compute_baseline:
    #     #     # train-test-split
    #     #     X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.3)
    #     #     start_time = time.time()
    #     #     mitigator = ExponentiatedGradient(clf, DemographicParity())
    #     #     mitigator.fit(X_train[:,:-1], y_train, sensitive_features=X_train[:,-1]) # instead of X_train and y_train
    #     #     y_pred_mitigated = mitigator.predict(X_test[:,:-1])
    #     #     ks_dict["fair_baseline"].append( unfairness(y_pred_mitigated[index_0], y_pred_mitigated[index_1]) )
    #     #     acc_dict["fair_baseline"].append( accuracy_score(y_test, y_pred_mitigated) )
    #     #     time_dict["fair_baseline"].append(time.time() - start_time)

    # if print_results:
    #     print()
    #     print_fairness_results(acc_dict, ks_dict, time_dict)
    
    # return acc_dict, ks_dict, time_dict, index_0, index_1, y_preds, y_probs, y_pred_fair_hard, y_pred_fair_soft, y_prob_fair_hard, y_prob_fair_soft

