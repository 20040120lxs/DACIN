import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from munkres import Munkres
import utilss


def best_map(L1, L2):
	#L1 should be the labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] =c[i]
	return newL2

def thrC(C, ro):
	if ro < 1:
		N = C.shape[1]
		Cp = np.zeros((N,N))
		S = np.abs(np.sort(-np.abs(C), axis=0))
		Ind = np.argsort(-np.abs(C), axis=0)
		for i in range(N):
			cL1 = np.sum(S[:,i]).astype(float)
			stop = False
			csum = 0
			t = 0
			while(stop == False):
				csum = csum + S[t,i]
				if csum > ro*cL1:
					stop = True
					Cp[Ind[0:t+1, i], i] = C[Ind[0:t+1, i], i]
				t = t + 1
	else:
		Cp = C
	return Cp

def post_proC(C, K, d, alpha):
	# C: coefficient matrix, K: number of clusters, d: dimension of each subspace
	n = C.shape[0]
	C = 0.5*(C + C.T)
	C = C - np.diag(np.diag(C)) + np.eye(n,n) # for sparse C, this step will make the algorithm more numerically stable
	r = d*K + 1
	U, S, _ = svds(C,r,v0 = np.ones(n))
	U = U[:,::-1]
	S = np.sqrt(S[::-1])
	S = np.diag(S)
	U = U.dot(S)
	U = normalize(U, norm='l2', axis = 1)
	Z = U.dot(U.T)
	Z = Z * (Z>0)
	L = np.abs(Z ** alpha)
	L = L/L.max()
	L = 0.5 * (L + L.T)
	spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize')
	spectral.fit(L)
	grp = spectral.fit_predict(L) + 1
	return grp, L

def err_rate(gt_s, s):
	c_x = best_map(gt_s,s)
	err_x = np.sum(gt_s[:] != c_x[:])
	missrate = err_x.astype(float) / (gt_s.shape[0])
	return missrate

def accuracy(gt_s, s):
	c_x = best_map(gt_s,s)
	accuracy=accuracy_score(gt_s, c_x)
	return accuracy

def precision(gt_s, s):
	c_x = best_map(gt_s,s)
	precision=precision_score(gt_s, c_x,average='weighted')
	return precision

def recall(gt_s, s):
	c_x = best_map(gt_s,s)
	recall=recall_score(gt_s, c_x,average='weighted')
	return recall

def f1(gt_s, s):
	c_x = best_map(gt_s,s)
	f1=f1_score(gt_s, c_x,average='weighted')
	return f1
def auc(gt_s, s,clusters):
	c_x = best_map(gt_s,s)
	c_x_one_hot = utilss.dense_to_one_hot(c_x, clusters)
	roc_auc = metrics.roc_auc_score(y_true=gt_s, y_score=c_x_one_hot, average='macro', multi_class='ovo')
	return roc_auc