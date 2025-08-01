import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

def make_unfair_classif(n = 500, n_features = 5, n_classes = 3, n_clusters = 2, p = 0.75):
    """
    the first half of classes will have p of probability having S = 1, the other half will have 1-p having S = 1
    """
    # rmks : all variances here are identities
    
    # centroids
    #centroids = np.random.uniform(-n_classes/2, n_classes/2, (n_classes, n_features))
    centroids = np.random.uniform(-1, 1, (n_classes, n_features))
    
    for cl in range(n_classes):
        
        # generate means based on centroids
        means = np.zeros((n_clusters, n_features))
        for comp in range(n_clusters):
            #means[comp] = centroids[cl] + (n_classes/3)*np.random.randn(n_features)
            means[comp] = centroids[cl] + np.random.randn(n_features)
            
        # Weight of each component, in this case all of them are equal
        weights = np.ones(n_clusters) / n_clusters
        
        # A stream of indices from which to choose the component
        mixture_idx = np.random.choice(n_clusters, size=n, replace=True, p=weights)
        
        # Data
        M = np.zeros((len(mixture_idx), n_features))
        for i in range(n_clusters):
            #M[mixture_idx == i] = np.random.multivariate_normal(means[i], (n_classes/3)*np.eye(n_features), sum(mixture_idx == i))
            M[mixture_idx == i] = np.random.multivariate_normal(means[i], np.eye(n_features), sum(mixture_idx == i))
        
        ym = cl*np.ones(len(mixture_idx))
        
        if cl == 0:
            M_all = M
            y_all = ym
        else:
            M_all = np.concatenate((M_all, M), axis = 0)
            y_all = np.concatenate((y_all, ym), axis = 0)
    
    # prepare contaminations
    M_all = np.concatenate((M_all, -1*np.ones((len(M_all), 1))), axis=1)
    
    # add contaminations(p) for the first half classes, and keep contaminations(1-p) for the rest
    for cl in np.arange(n_classes):
        if cl < n_classes//2:
            index = (y_all == cl)
            n_index = np.sum(index)
            M_all[index, -1] = 2*np.random.binomial(1, p, n_index)-1
        else:
            index = (y_all == cl)
            n_index = np.sum(index)
            M_all[index, -1] = 2*np.random.binomial(1, 1-p, n_index)-1
    
    # shuffle
    shuffle_indexes = np.arange(len(y_all))
    np.random.shuffle(shuffle_indexes)
    M_all = M_all[shuffle_indexes]
    y_all = y_all[shuffle_indexes]
        
    return(M_all, y_all)


def make_unfair_poolclassif(n, n_features, n_classes, n_clusters, n_pool = 0, p = 0.7):
    """
    make unfair classification dataset (labeled dataset) with pool data (unlabeled dataset)
    """
    X, y = make_unfair_classif(n = n + n_pool, n_features = n_features, n_classes = n_classes, n_clusters = n_clusters, p = p)
    if n_pool > 0:
        X_pool = X[:n_classes*n_pool]
    else:
        X_pool = None
    X = X[n_classes*n_pool:]
    y = y[n_classes*n_pool:]
    
    return(X, y, X_pool)


def data_viz(X, y, S = None, alpha = 0.9):
    """
    visualize the first two dimensions of the dataset along with their label and contamination
    """
    sns.set_palette("colorblind", 10)
    palette = sns.color_palette("colorblind", 10)
    if S is None:
        for cl in np.unique(y):
            plt.scatter(X[y==cl,0], X[y==cl,1], alpha = alpha, marker = "o")
    else:
        for cl in np.unique(y):
            plt.scatter(X[(y==cl) & (S!=1),0], X[(y==cl) & (S!=1),1], alpha = (alpha)/1.2, marker = "+", label="S=-1", color=palette[int(cl)])
            plt.scatter(X[(y==cl) & (S==1),0], X[(y==cl) & (S==1),1], alpha = alpha, marker = "^", label="S=+1", color=palette[int(cl)])
        plt.legend()
    #if S is not None:
    #    plt.scatter(X[S == 1,0], X[S == 1,1], alpha = alpha/3, color="red", marker="^")
    plt.show()
    return None

def data_viz_tsne(X, y):
    """
    visualize the first two dimensions of the dataset along with their label and contamination
    """
    X_reduc = X[:500,]
    y_reduc = y[:500]
    X_embedded = TSNE(n_components=2).fit_transform(X_reduc)
    data_viz(X_embedded, y_reduc, X_reduc[:,-1])
    return None