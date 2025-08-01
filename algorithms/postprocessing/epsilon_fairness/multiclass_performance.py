from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def get_multiclass_performance(model, X_train, X_test, y_train, y_test, print_results = True):

    # define the ovr strategy
    ovr = OneVsRestClassifier(model)

    # fit model
    ovr.fit(X_train, y_train)

    # make predictions
    yhat_proba = ovr.predict_proba(X_test) # not a softmax, just a normalization on the positives probs after OvR
    yhat = ovr.predict(X_test)

    # scores
    score_test = accuracy_score(y_test, ovr.predict(X_test))
    score_train = accuracy_score(y_train, ovr.predict(X_train))
    
    if print_results:
        print("test :", score_test)
        print("train :", score_train)
        C = confusion_matrix(y_test, yhat)
        plot_confusion_matrix(C, classes = np.arange(ovr.n_classes_))
    
    return score_test, score_train, ovr