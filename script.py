# Authors: Bellet, Gramfort, Salmon

from time import time

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.mldata import fetch_mldata

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
plt.style.use('ggplot')


####################################################################
# Download the data (if not present); load it as numpy arrays
dataset_name = 'covtype.binary'
covtype = fetch_mldata(dataset_name)
covtype.data = covtype.data.toarray()  # convert to dense

####################################################################
# Extract features
X_train, X_test, y_train, y_test = \
    train_test_split(covtype.data[:50000, :], covtype.target[:50000],
                     train_size=10000, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

####################################################################
# SVM classfication (Question 1)

from sklearn.svm import SVC, LinearSVC

print("Fitting SVC rbf on %d samples..." % X_train.shape[0])
t0 = time()
# TODO
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))

print("Predicting with SVC rbf on %d samples..." % X_test.shape[0])
t1 = time()
# TODO
accuracy_kernel=clf.score(X_test,y_test)
print("done in %0.3fs" % (time() - t1))
timing_kernel = time() - t0
print("classification accuracy: %0.3f" % accuracy_kernel)
print 'timing_kernel: ',str(timing_kernel)

# TODO with LinearSVC
print("Fitting SVC linear on %d samples..." % X_train.shape[0])
t0 = time()
# TODO
clf = LinearSVC(dual=False)
clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))

print("Predicting with SVC linear on %d samples..." % X_test.shape[0])
t1 = time()
# TODO
accuracy_kernel=clf.score(X_test,y_test)
print("done in %0.3fs" % (time() - t1))
timing_linear = time() - t0
print("classification accuracy: %0.3f" % accuracy_kernel)
print'timing_linear: ',str(timing_linear)
####################################################################
# Gram approximation
from TP_kernel_approx_source import rank_trunc, nystrom, random_features

p = 200
r_noise = 100
r_signal = 20

intensity = 50

rng = np.random.RandomState(42)
X_noise = rng.randn(r_noise, p)
X_signal = rng.randn(r_signal, p)

gram_signal = np.dot(X_noise.T, X_noise) + intensity * np.dot(X_signal.T,
                                                              X_signal)
n_ranks = 100
ranks = np.arange(1, n_ranks + 1)
timing_fast = np.zeros(n_ranks)
timing_slow = np.zeros(n_ranks)
accuracy = np.zeros(n_ranks)

# TODO : Question 2 Implement rank_trunc function in source file.
# TODO : Question 3 Evaluate accuracy with Frobenius norm as a function
# of the rank for both svd solvers

# Use linalg.norm(A, 'fro') to compute Frobenius norm of A
for k, rank in enumerate(ranks):
    tf=time()
    gk,_,_=rank_trunc(gram_signal,rank)
    timing_fast[k]=time()-tf
    tsl=time()
    gks,_,_=rank_trunc(gram_signal,rank,fast=False)
    timing_slow[k]=time()-tsl
    accuracy[k]= linalg.norm(gram_signal-gk,ord='fro')/linalg.norm(gram_signal,ord='fro')
#    accuracy[k]= linalg.norm(gram_signal-gks,ord='fro')/linalg.norm(gram_signal,ord='fro')

print("done q23")
#plt.plot(accuracy,'o-')
#plt.title('Accuracy of approximasion by truncated SVD as a function of rank')
#plt.figure(2)
#plt.plot(ts,'o-')
#plt.title('Computation time using fast/slow truncSVD as a function of rank')

#plt.savefig('q2.png')
####################################################################
# Display
opt={'figroot':'report/figs/'}

fig, axes = plt.subplots(ncols=1, nrows=2)
ax1, ax2 = axes.ravel()

ax1.plot(ranks, timing_fast, '-')
ax1.plot(ranks, timing_slow, '-')

ax1.set_xlabel('Rank')
ax1.set_ylabel('Time')
ax1.legend({'timing fast','timing slow'},loc=2)
ax2.plot(ranks, accuracy, '-')
ax2.set_xlabel('Rank')
ax2.set_ylabel('Accuracy')
# plt.tight_layout()
plt.draw()
plt.savefig(opt['figroot']+'q23-new.png')
# plt.show()

####################################################################
# Random Kernel Features:

n_samples, n_features = X_train.shape
n_samples_test, _ = X_test.shape
gamma = 1. / n_features

# TODO : Question 4 Implement random features in source file.

Z_train, Z_test = random_features(X_train, X_test, gamma, c=300, seed=44)

# TODO : Question 5 Estimate training, testing time and accuracy
print("Fitting SVC linear on %d samples..." % n_samples)
t0 = time()
# TODO
clf = LinearSVC(dual=False)
clf.fit(Z_train, y_train)
print("Q5-train done in %0.3fs" % (time() - t0))

print("Predicting with SVC rbf on %d samples..." % n_samples_test)
t1 = time()
# TODO
accuracy_kernel=clf.score(Z_test,y_test)
print("done in %0.3fs" % (time() - t1))
print("classification accuracy: %0.3f" % accuracy_kernel)
print'Q5-training-testing time: ',str(time() - t0)

####################################################################
# SVM Nystrom:

# TODO : Question 6-7 Implement nystrom in source file.

Z_train, Z_test = nystrom(X_train, X_test, gamma, c=500, k=200, seed=44)

print("Fitting SVC linear on %d samples..." % n_samples)
t0 = time()
clf = LinearSVC(dual=False)
clf.fit(Z_train, y_train)
print("fit done in %0.3fs" % (time() - t0))

print("Predicting with SVC linear on %d samples..." % n_samples_test)
t1 = time()
accuracy = clf.score(Z_test, y_test)
print("pred done in %0.3fs" % (time() - t1))
print("classification accuracy: %0.3f" % accuracy)
print'Q7-training-testing time: ',str(time() - t0)

####################################################################
# Results / comparisons:

ranks = list(range(20, 500, 50))
n_ranks = len(ranks)
timing_rkf = np.zeros(n_ranks)
timing_nystrom = np.zeros(n_ranks)

accuracy_nystrom = np.zeros(n_ranks)
accuracy_rkf = np.zeros(n_ranks)

for i, c in enumerate(ranks):
    print 'task-',str(i),'--- c=',str(c)
    t0 = time()
    # TODO Question 7/// 8 en fait
    Z_train, Z_test = random_features(X_train, X_test, gamma, c=c, seed=44)
    print 'shape of Ztrain: ', str(Z_train.shape)
    clf = LinearSVC(dual=False)
    clf.fit(Z_train, y_train)
    accuracy_rkf[i] = clf.score(Z_test,y_test)
    timing_rkf[i] = time() - t0
    t1=time()
    Z_trainn, Z_testn = nystrom(X_train, X_test, gamma, c=c,k=c-10, seed=44)
    print 'shape of Ztrain_nystrom: ',str(Z_trainn.shape)
    clf = LinearSVC(dual=False)
    clf.fit(Z_trainn, y_train)
    accuracy_nystrom[i] = clf.score(Z_testn,y_test)
    timing_nystrom[i] = time() - t1

fig, axes = plt.subplots(ncols=1, nrows=2)
ax1, ax2 = axes.ravel()

ax1.plot(ranks, timing_rkf, 'o-')
ax1.plot(ranks, timing_nystrom, '*-')

ax1.set_xlabel('c')
ax1.set_ylabel('Time')
ax1.legend(['SVM with RKF','SVM with Nystrom'],loc=2)
ax2.plot(ranks, accuracy_rkf, 'o-')
ax2.plot(ranks,accuracy_nystrom,'*-')

ax2.set_xlabel('Rank')
ax2.set_ylabel('Accuracy')
# plt.tight_layout()
plt.draw()
plt.savefig(opt['figroot']+'q8-new.png')
# plt.show()

####################################################################
# Display bis

# TODO : Question 8 synthetize results
