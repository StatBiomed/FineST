## 2024.10.31 Create by Tianjie Wang (our co-author)
##            https://github.com/jackywangtj66/SparseAEH
## 2024.10.31 intergrate base.py and plot.py together

import numpy as np
#from function import RBF_kernel
import scipy
import time
from sklearn.metrics.pairwise import laplacian_kernel,rbf_kernel
from scipy.sparse import csr_matrix
from operator import itemgetter
import warnings
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import random

def _pinv_1d(v, eps=1e-5):
    return np.array([0 if abs(x) <= eps else 1/x for x in v], dtype=float)

def _pexp(array):
    return np.array([np.inf if n>50 else (np.exp(n) if n >-50 else 0) for n in array])


class Kernel:

    def __init__(self,spatial,ss_loc=None,group_size=16,cov=None,dependency=None,
                 d=5,kernel='laplacian',l=0.01):
        """
        number of superspots: M
        cov: full pre-determined covariance matrix
        dependency format: M-length list of lists, 
        the i-th element indicates the superspots that the i-th superspot is dependent on
        ss_loc: M-length list of lists, the i-th element indicates the spots (ordinal) that 
        the i-th superspots contain spatial: spatial coordinates of all spots
        l: hyperparameter for kernel
        """
        super().__init__()
        self.N = len(spatial)
        #self.all_cov = [[0 for _ in range(self.M)] for _ in range(self.M)]
        #self.ss_loc = ss_loc
        self.spatial = spatial
        self.cond_mean = None
        self.l = l
        self.d = d
        self.A = []
        self.cond_cov = []
        self.kernel = kernel
        self._initialize(cov,ss_loc,dependency,group_size)
    
    def get_mat(self,rows,cols):
        row = []
        for r in rows:
            col = []
            for c in cols:
                if r >= c:
                    col.append(self.all_cov[r][c])
                else:
                    col.append(self.all_cov[c][r].T)
            row.append(np.hstack(col))
        return np.vstack(row)

    
    def _initialize(self,cov,ss_loc,dependency,group_size):
        self._init_ss(ss_loc,dependency,group_size)
        self._init_ds_loc()
        self._init_all_cov(cov)
        self._init_ds_eig()
        self._init_base_cond_cov()        
        #self._init_cond_cov()

    def _init_ss(self,ss_loc,dependency,group_size):
        if ss_loc is not None:
            self.ss_loc = ss_loc
            self.dependency = dependency
            self.M = len(dependency)
        else:
            centers = []
            self.ss_loc = []
            group = self.N / group_size
            g_r = int(np.sqrt(group))
            g_c = int(group/g_r)
            #self.M = g_c * g_r
            kmeans_r = KMeans(n_clusters=g_r, random_state=0, n_init=10)
            kmeans_c = KMeans(n_clusters=g_c, random_state=0, n_init=10)
            kmeans_r.fit(self.spatial[:,0:1])
            kmeans_c.fit(self.spatial[:,1:])
            for r in range(g_r):
                for c in range(g_c):
                    pos = np.logical_and(kmeans_r.labels_ == r, kmeans_c.labels_ == c)
                    if pos.any():
                        centers.append(np.average(self.spatial[pos],axis=0))
                        self.ss_loc.append(np.arange(len(self.spatial))[pos])
            #print(self.ss_loc)
            #print(centers)
            centers = np.vstack(centers)
            self.M = centers.shape[0]
            self.dependency = [[] for _ in range(self.M)]
            distance = distance_matrix(centers,centers)
            for i in range(self.M):
                if i <= self.d:
                    self.dependency[i] = list(range(i))
                else:
                    #self.dependency[i] = np.argmax(distance[i,:i],self.d)
                    self.dependency[i] = distance[i,:i].argsort()[:self.d]


    def _init_ds_loc(self):
        self.ds_loc = []        
        for i,ds in enumerate(self.dependency):
           ind = []
           for ss in ds:
               ind += np.array(self.ss_loc[ss]).tolist()
           self.ds_loc.append(ind)
    
    def _init_all_cov(self,cov):
        self.all_cov = [[0 for _ in range(self.M)] for _ in range(self.M)]
        for i in range(self.M):
            if cov is not None:
                self.all_cov[i][i] = cov[np.ix_(self.ss_loc[i],self.ss_loc[i])]
            else:
                if self.kernel == 'rbf':
                    self.all_cov[i][i] = rbf_kernel(self.spatial[self.ss_loc[i]],gamma=self.l)
                else:
                    self.all_cov[i][i] = laplacian_kernel(self.spatial[self.ss_loc[i]],gamma=self.l)
            for j in self.dependency[i]:
                if cov is not None:
                    self.all_cov[i][j] = cov[np.ix_(self.ss_loc[i],self.ss_loc[j])]
                else:
                    if self.kernel == 'rbf':
                        self.all_cov[i][j] = rbf_kernel(self.spatial[self.ss_loc[i]],
                                                        self.spatial[self.ss_loc[j]],gamma=self.l)
                    else:
                        self.all_cov[i][j] = laplacian_kernel(self.spatial[self.ss_loc[i]],
                                                              self.spatial[self.ss_loc[j]],gamma=self.l)
        for k in range(self.M):
            for i in self.dependency[k]:
                for j in self.dependency[k]:
                    if j<i and isinstance(self.all_cov[i][j],int):
                        if cov is not None:
                            self.all_cov[i][j] = cov[np.ix_(self.ss_loc[i],self.ss_loc[j])]
                        else:
                            if self.kernel == 'rbf':
                                self.all_cov[i][j] = rbf_kernel(self.spatial[self.ss_loc[i]],
                                                                self.spatial[self.ss_loc[j]],gamma=self.l)
                            else:
                                self.all_cov[i][j] = laplacian_kernel(self.spatial[self.ss_loc[i]],
                                                                      self.spatial[self.ss_loc[j]],
                                                                      gamma=self.l)
    

    def _init_ds_eig(self):
        #C_m,C_m
        self.ds_eig = []
        for i in range(self.M):
            if len(self.dependency[i]) == 0:
                self.ds_eig.append(())
                self.A.append(())            
            else:            
                ds_cov = self.get_mat(self.dependency[i],self.dependency[i])
                s,u = np.linalg.eigh(ds_cov)
                #s_inv = _pinv_1d(s)
                self.ds_eig.append((s,u))
                self.A.append(self.get_mat([i],self.dependency[i]) @ u)
    

    def _init_base_cond_cov(self):
        for i in range(self.M):
            if len(self.dependency[i]) == 0:
                self.cond_cov.append(self.get_mat([i],[i]))
            else:
                self.cond_cov.append(self.get_mat([i],[i]) - np.multiply(1/self.ds_eig[i][0],self.A[i])@self.A[i].T)
    

    def update_cond_cov(self,Delta):
        cond_cov_eig = [[] for _ in range(len(Delta))]
        for eig, delta in zip(cond_cov_eig,Delta):
            for i in range(self.M):
                if len(self.dependency[i]) == 0:
                #self.cond_cov.append(self.kernel.base_cond_cov[i]+self.delta*np.eye(len(self.kernel.ss_loc[i])))
                    s,u = np.linalg.eigh(self.cond_cov[i]+delta*np.eye(len(self.ss_loc[i])))
                else:
                    s,u = np.linalg.eigh(self.cond_cov[i]+delta*np.eye(len(self.ss_loc[i]))+
                                         delta*np.multiply(1/((self.ds_eig[i][0]+delta)*self.ds_eig[i][0]),self.A[i])@self.A[i].T)
                eig.append((s,u))
        return cond_cov_eig

class MixedGaussian:
    def __init__(self,spatial,ss_loc=None,group_size=16,
                 cov=None,dependency=None,d=5,kernel='rbf',l=0.01):
        self.kernel = Kernel(spatial,ss_loc,group_size,cov,dependency,d,kernel,l)

    def update_cond_mean(self):
        #Y:N*G  mean:N*K
        self.cond_dev = self.Y[np.newaxis,:] - self.mean.transpose()[:,:,np.newaxis]   #K,N,G
        dev = self.cond_dev.copy()
        for i in range(self.kernel.M):
                if len(self.kernel.dependency[i]) > 0:
                    for k in range(self.K):
                        self.cond_dev[k,self.kernel.ss_loc[i],:] = self.cond_dev[k,self.kernel.ss_loc[i],:] - \
                        np.multiply(1/(self.kernel.ds_eig[i][0]+self.delta[k]),
                                    self.kernel.A[i]) @ self.kernel.ds_eig[i][1].T @ dev[k,self.kernel.ds_loc[i],:]

    def compute_ll(self,cond_cov_eig):
        ll = np.zeros((self.G,self.K))
        for k in range(self.K):
            ll[:,k] = np.log(2 * np.pi)*self.N + 2*np.log(self.sigma_sq[k])*self.N
            for i in range(self.kernel.M):
                det = np.prod(cond_cov_eig[k][i][0])
                if det <= 0:
                    print(cond_cov_eig[k][i][0]) 
                ll[:,k] += np.log(det)
                temp = self.cond_dev[k][self.kernel.ss_loc[i],:].T @ cond_cov_eig[k][i][1]
                ll[:,k] += np.sum(np.multiply(1/cond_cov_eig[k][i][0],np.square(temp)),
                                  axis=1)/self.sigma_sq[k]
        ll = ll*-0.5
        return ll
    
    def update_param(self,omega):
        new_mean= np.zeros_like(self.mean)
        #mean
        for k in range(self.K):
            new_mean[:,k:(k+1)] = self.Y @ omega[:,k:(k+1)] / np.sum(omega[:,k])
        new_dev = self.Y[np.newaxis,:] - new_mean.transpose()[:,:,np.newaxis]
        #pi
        if self.update_pi:
            self.pi = np.average(omega,axis=0)
        
        for k in range(self.K):
            self.cov_new = []
            for i in range(self.kernel.M):
                l = len(self.kernel.ss_loc[i])
                cov_i = np.zeros((l,l))
                for g in range(self.G):
                    cov_i += omega[g,k]*np.outer(new_dev[k,self.kernel.ss_loc[i],g],
                                                 new_dev[k,self.kernel.ss_loc[i],g])
                cov_i = cov_i/np.sum(omega[:,k])
                self.cov_new.append(cov_i)
            numer,t_2 = 0,0
            for i in range(self.kernel.M):
                numer += np.sum(np.multiply(self.kernel.all_cov[i][i],self.cov_new[i]))
                #denom += np.sum(np.multiply(self.kernel.all_cov[i][i],self.kernel.all_cov[i][i]))
                #t_1 += np.trace(self.kernel.all_cov[i][i])
                t_2 += np.trace(self.cov_new[i])
            #print(numer,denom,t_1,t_2)
            self.sigma_sq[k] = (numer - self.t_1*t_2/self.N)*0.5 / (self.denom - self.t_1**2/self.N) + self.sigma_sq[k]/2
            #print(self.sigma_sq[k],(numer - t_1*t_2/self.N) / (denom - t_1**2/self.N))
            if self.sigma_sq[k] == 0:
                self.delta[k] = t_2*0.5/self.N + self.delta[k]/2
            else:
                self.delta[k] = (t_2-self.sigma_sq[k]*self.t_1)*0.5/(self.N*self.sigma_sq[k]) + self.delta[k]/2
            if self.delta[k]<=0:
                self.delta[k] = 0 
                self.sigma_sq[k] = numer / self.denom 
        return new_mean
    
    def update_mean(self,omega):
        new_mean= np.zeros_like(self.mean)
        #mean
        for k in range(self.K):
            new_mean[:,k:(k+1)] = self.Y @ omega[:,k:(k+1)] / np.sum(omega[:,k])
        new_dev = self.Y[np.newaxis,:] - new_mean.transpose()[:,:,np.newaxis]
        #pi
        if self.update_pi:
            self.pi = np.average(omega,axis=0)
        return new_mean
    
    def param_init(self):
        samp_ind = random.sample(range(self.G),self.G//10)
        sample = self.Y[:,samp_ind]
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(sample.T)
        return kmeans.cluster_centers_.T

    def run_cluster(self,Y,K,pi=None,mean=None,sigma_sq=None,delta=None,iter=500,
                    threshold=5e-2,init_mean='k_means',update_pi=True):
        self.Y = Y
        self.K = K
        self.N,self.G = self.Y.shape
        self.update_pi = update_pi
        if pi is not None:
            self.pi = np.array(pi,dtype=float)
        else:
            #self.pi = np.random.dirichlet(np.ones(self.K))    
            self.pi = np.ones(self.K,dtype=float) /self.K
        if mean is not None:
            self.mean = np.array(mean,dtype=float)
        else:    
            #self.mean = np.abs(np.random.normal(size=(self.N, self.K)))
            self.mean = np.random.uniform(size=(self.N, self.K))
            if init_mean == 'k_means':
                self.init_mean = self.param_init()
                self.mean = self.init_mean
            elif init_mean == 'sample':
                self.init_mean = self.Y[:,np.random.choice(self.G,self.K)]
            elif isinstance(init_mean,np.ndarray):
                self.init_mean = init_mean
            
        #return self.mean
        if sigma_sq is not None:
            self.sigma_sq = np.array(sigma_sq,dtype=float)
        else:
            self.sigma_sq = np.ones(self.K,dtype=float)*0.1
        if delta is not None:
            self.delta = np.array(delta,dtype=float)
        else:
            self.delta = np.ones(self.K,dtype=float)*1
        
        # power = np.zeros((G,self.K))
        self.omega = np.ones((self.G,self.K))/self.K
        converge = False
        count = 0
        #self.ll = np.zeros((self.G,self.K))
        self.denom,self.t_1 = 0,0
        for i in range(self.kernel.M):
            self.denom += np.sum(np.multiply(self.kernel.all_cov[i][i],self.kernel.all_cov[i][i]))
            self.t_1 += np.trace(self.kernel.all_cov[i][i])
        
        while not converge:
            print('Iteration {}'.format(count))
            cond_cov_eig = self.kernel.update_cond_cov(self.delta)
            self.update_cond_mean()

            self.ll = self.compute_ll(cond_cov_eig)
            #print(self.ll)
            #print(compute_likelihood(self.Y,self.kernel,self.cond_dev[0],self.sigma_sq[0],cond_cov_eig[0]))
            for k in range(self.K):
                if self.pi[k] == 0:
                    self.omega[:,k] = 0
                else:
                    for g in range(self.G):
                        #omega[g,k] = 3/4*self.pi[k]/np.sum(self.pi * _pexp((self.ll[g]-self.ll[g][k])/np.sqrt(self.N))) + omega[g,k]/4
                        self.omega[g,k] = self.pi[k]/np.sum(self.pi * _pexp((self.ll[g]-self.ll[g][k])/np.sqrt(self.N))) + 1e-3/self.G
                        # if np.sum(self.pi * _pexp((ll[g]-ll[g][k])/np.sqrt(self.N))) == 0:
                        #     print(ll[g],ll[g][k])

            #print(self.omega)
            new_mean = self.update_mean(self.omega)
            #print(self.delta,self.sigma_sq)
            count += 1
            if count > iter or np.mean(np.abs(new_mean-self.mean))<0.1:
                converge = True
            self.mean = new_mean
            #converge = True
            #print(self.pi,self.sigma_sq,self.delta)
            #indexes = np.array([np.arange(0,2),np.arange(300,302),np.arange(800,802)])
            #print(self.sigma_sq,self.delta,self.pi,omega[indexes])
        #return self.mean
        print('updating variance')
        converge = False
        while not converge:
            print('Iteration {}'.format(count))
            cond_cov_eig = self.kernel.update_cond_cov(self.delta)
            self.update_cond_mean()

            self.ll = self.compute_ll(cond_cov_eig)
            #print(self.ll)
            #print(compute_likelihood(self.Y,self.kernel,self.cond_dev[0],self.sigma_sq[0],cond_cov_eig[0]))
            for k in range(self.K):
                if self.pi[k] == 0:
                    self.omega[:,k] = 0
                else:
                    for g in range(self.G):
                        #omega[g,k] = 3/4*self.pi[k]/np.sum(self.pi * _pexp((self.ll[g]-self.ll[g][k])/np.sqrt(self.N))) + omega[g,k]/4
                        self.omega[g,k] = self.pi[k]/np.sum(self.pi * _pexp((self.ll[g]-self.ll[g][k]))) + 1e-3/self.G
                        # if np.sum(self.pi * _pexp((ll[g]-ll[g][k]))) == 0:
                        #     print(ll[g],ll[g][k])

            #print(self.omega)
            new_mean = self.update_param(self.omega)
            #print(self.delta,self.sigma_sq)
            count += 1
            if count > iter or np.mean(np.abs(new_mean-self.mean))<threshold:
                converge = True
            self.mean = new_mean
            #print(self.pi,self.sigma_sq,self.delta)
        self.labels = np.argmax(self.omega,axis=1)    
        return self.mean

    def cluster_counts(self,query_label=None):
        if query_label is not None:
            return np.sum(self.labels==query_label)
        else:
            return np.array([np.sum(self.labels==i) for i in range(self.K)])