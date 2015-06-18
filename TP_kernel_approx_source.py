# Authors: Bellet, Gramfort, Salmon

from math import sqrt
import numpy as np

from scipy.sparse.linalg import svds
from scipy.linalg import svd

from sklearn.metrics.pairwise import rbf_kernel


def rank_trunc(gram_mat, k, fast=True):
    """
    k-th order approximation of the Gram Matrix G.

    Parameters
    ----------
    gram_mat : array, shape (n_samples, n_samples)
        the Gram matrix
    k : int
        the order approximation
    fast : bool
        use svd (if False) or svds (if True).

    Return
    ------
    gram_mat_k : array, shape (n_samples, n_samples)
        The rank k Gram matrix.
    """
    if fast:
        u,s,v=svds(gram_mat,k)
        #pass  # TODO
    else:
        U,S,V=svd(gram_mat) #full by default--> both U,V: [nxn] here
        s=S[:k]
        u=U[:k,:k]
        v=V[:k,:k]
        #pass  # TODO
    gram_mat_k = (u.dot(np.diag(s))).dot(v)
    return gram_mat_k, u, s


def random_features(X_train, X_test, gamma, c=300, seed=44):
    """Compute random kernel features

    Parameters
    ----------
    X_train : array, shape (n_samples1, n_features)
        The train samples.
    X_test : array, shape (n_samples2, n_features)
        The test samples.
    gamma : float
        The Gaussian kernel parameter
    c : int
        The number of components
    seed : int
        The seed for random number generation

    Return
    ------
    X_new_train : array, shape (n_samples1, c)
        The new train samples.
    X_new_test : array, shape (n_samples2, c)
        The new test samples.
    """
    rng = np.random.RandomState(seed)
    # TODO Question 4
    [n1,p]=X_train.shape
    n2=X_test.shape[0]
    W=sqrt(2.*gamma)*rng.randn(p,c)
    b=rng.uniform(0,2*np.pi,size=c)
    X_new_train = sqrt(2./c)*np.cos(X_train.dot(W)+np.outer(np.ones(n1),b))
    X_new_test = sqrt(2./c)*np.cos(X_test.dot(W)+np.outer(np.ones(n2),b))
    return X_new_train, X_new_test


def nystrom(X_train, X_test, gamma, c=500, k=200, seed=44):
    """Compute nystrom kernel approximation

    Parameters
    ----------
    X_train : array, shape (n_samples1, n_features)
        The train samples.
    X_test : array, shape (n_samples2, n_features)
        The test samples.
    gamma : float
        The Gaussian kernel parameter
    c : int
        The number of points to sample for the approximation
    k : int
        The number of components
    seed : int
        The seed for random number generation

    Return
    ------
    X_new_train : array, shape (n_samples1, c)
        The new train samples.
    X_new_test : array, shape (n_samples2, c)
        The new test samples.
    """
    rng = np.random.RandomState(seed)
    # TODO Question 6
    [n1,p] = X_train.shape
    n2 = X_test.shape[0]

    I=rng.randint(n1,size=c)
    G=rbf_kernel(X_train[I],X_train[I])
    Gk,uk,sk=rank_trunc(G,k) #fast=True
    Mk= uk.dot(np.diag(np.sqrt(1./sk)))

    Ttr = rbf_kernel(X_train, X_train[I])
    Tte = rbf_kernel(X_test,X_train[I])
    X_new_train =  Ttr.dot(Mk)
    X_new_test = Tte.dot(Mk)

    return X_new_train, X_new_test
