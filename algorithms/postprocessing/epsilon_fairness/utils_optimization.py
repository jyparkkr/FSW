import numpy as np
from numpy.random import multivariate_normal
from scipy.optimize import minimize, LinearConstraint


def optSCIPY(fun, n_classes):
    lam0 = np.zeros(n_classes)
    res = minimize(fun, lam0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
    lam = res.x
    return lam

def optSCIPY_bivar(fun, n_classes):
    linear_constraint = LinearConstraint(np.eye(2*n_classes), 0, np.inf)
    lam0 = np.zeros(2*n_classes)
    res = minimize(
        fun,
        lam0,
        constraints=[linear_constraint],
        method='trust-constr')
    lam = res.x
    return lam[:n_classes], lam[n_classes:]

def optSCIPY_bivar_bis(fun, n_classes):
    ineq_cons = {
        'type': 'ineq',
        'fun' : lambda x: x}
    lam0 = np.zeros(2*n_classes)
    res = minimize(
        fun,
        lam0,
        constraints=[ineq_cons],
        options={'ftol': 1e-9, 'disp': False},
        method='SLSQP')
    lam = res.x
    return lam[:n_classes], lam[n_classes:]

def optSAGD(fun, n_classes, c = 0.1, T = 1000):
    """
    Smoothed Accelerated Gradient Descent
    """
    lam = np.zeros((T+1, n_classes))
    zs = np.zeros((T+1, n_classes))
    tau = np.zeros(T)
    gam = np.zeros(T)

    for t in np.arange(1, T):
        tau[t] = ( 1 + np.sqrt(1+4*tau[t-1]**2) ) / 2
        gam[t] = (1 - tau[t-1]) / tau[t]

        gradient = fun(lam[t,:], c = c)

        zs[t+1,:] = lam[t,:] - (c/2) * gradient
        lam[t+1,:] = (1 - gam[t]) * zs[t+1,:] + gam[t]*zs[t,:]

    return(lam[-1,:])

#def optSAGD_bivar(fun1, fun2, n_classes, epsilon = 0.1, c = 0.1, T = 1000):
#    """
#    Smoothed Accelerated Gradient Descent
#    """
#    lam = np.zeros((T+1, n_classes))
#    beta = np.zeros((T+1, n_classes))
#    zs1 = np.zeros((T+1, n_classes))
#    zs2 = np.zeros((T+1, n_classes))
#    tau = np.zeros(T)
#    gam = np.zeros(T)
#
#    for t in np.arange(1, T):
#        tau[t] = ( 1 + np.sqrt(1+4*tau[t-1]**2) ) / 2
#        gam[t] = (1 - tau[t-1]) / tau[t]
#
#        gradient1 = fun1(lam[t,:], beta[t,:], epsilon = epsilon, c = c)
#        gradient2 = fun2(lam[t,:], beta[t,:], epsilon = epsilon, c = c)
#        
#        zs1[t+1,:] = lam[t,:] - (c/2) * gradient1
#        zs2[t+1,:] = beta[t,:] - (c/2) * gradient2
#        lam[t+1,:] = (1 - gam[t]) * zs1[t+1,:] + gam[t] * zs1[t,:]#np.maximum((1 - gam[t]) * zs1[t+1,:] + gam[t] * zs1[t,:], 0)#
#        beta[t+1,:] = (1 - gam[t]) * zs2[t+1,:] + gam[t] * zs2[t,:]#np.maximum((1 - gam[t]) * zs2[t+1,:] + gam[t] * zs2[t,:], 0)#
#        #lam[t+1,:] = np.maximum((1 - gam[t]) * zs1[t+1,:] + gam[t] * zs1[t,:], 0)
#        #beta[t+1,:] = np.maximum((1 - gam[t]) * zs2[t+1,:] + gam[t] * zs2[t,:], 0)
#    return(lam[-1,:], beta[-1,:])

def optCE(fun, n = 1000, d = 3, eps = 1e-8, max_iter = 1000, tau = 0.01, print_results = True):
    """
    this function compute the arg max of the function 'fun' by cross-entropy method
    """
    mu = np.random.rand(d)
    sigma = np.eye(d)
    
    t = 0
    while True:
        t += 1
        if print_results:
            if t%10==0:
                print("ite :", t)
                print(mu)
                print(np.max(sigma))
                print()
        
        # step 1 : sample Yi
        Y = multivariate_normal(mean = mu, cov = sigma, size=n)

        # step 2 : pick the top Yi that minimize the function
        samples = - np.array(list(map(fun, Y)))
        n_top = round(tau * n)
        top_ind = np.argsort(samples)[::-1][:n_top]
        Y_top = Y[top_ind, :]

        # step 3 : estimate by MLE mu and sigma
        mu = np.mean(Y_top, 0)
        sigma = np.cov(Y_top.T)
            
        # step 4 : stopping criter
        if np.max(sigma) < eps or t >= max_iter:
            if print_results:
                print(t)
                print(mu)
            break
    return mu