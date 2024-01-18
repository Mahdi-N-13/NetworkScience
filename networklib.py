import numpy as np
# import matplotlib.pyplot as plt
# from random import randint,sample,random as rnd
# import networkx as nx
from tqdm.autonotebook import tqdm
from scipy.stats import linregress
# from time import time as timer
#%%
def raise_error_network(n,m,func_name):
    if n <=0 :
        print(f"n must be positive number in "+func_name)
        return True
    elif m <= 0:
        print(f"m must be positive number in "+func_name)
        return True
    elif m >= n:
        print(f"m must be lower than n in "+func_name)
        return True
#%%
def histo_int(z,nbins=None,logbins=True,density=True,return_log_linregress=True,start_end_fit=None):
    if logbins:
        if nbins is None:
            nbins = 50
        xbins = np.unique(np.logspace(0,np.log10(z.max()+2),nbins).astype(int))
    else:
        xbins = np.arange(z.min(),z.max()+2)
    y = np.histogram(z,xbins,density=density)[0]
    nz = y!=0
    if return_log_linregress:
        x = xbins[:-1][nz]
        y = y[nz]
        x_log = np.log(x)
        y_log = np.log(y)
        # N = len(x)
        # D = np.zeros(N)
        # for st in range(N):
        #     landa = 1 + N/(np.log(x/(st-0.5)).sum())
        #     slope = linregress(x_log[st:],y_log[st:]).slop
        try:
            if len(start_end_fit) == 1:
                st = start_end_fit
                en = len(x)
            elif len(start_end_fit) == 2:
                st,en = start_end_fit
            else:
                st = 0
                en = len(x)
        except:
            st = 0
            en = len(x)
        return x,y,linregress(np.log(x[st:en]),np.log(y[st:en]))
    
    else:
        return xbins[:-1][nz],y[nz]
def histo_list(Z,nbins=50,logbins=True,density=True,nonezero=True):
    zmax = max([z.max() for z in Z])+2
    zmin = min([z.min() for z in Z])
    if logbins:
        if zmin < 1:
            zmin = 0
        else:
            zmin = np.log10(zmin)
        xbins = np.unique(np.logspace(zmin,np.log10(zmax),nbins).astype(int))
    else:
        xbins = np.arange(zmin,zmax)
    Y = [np.histogram(z,xbins,density=density)[0] for z in Z]
    x = xbins[:-1]
    if nonezero:
        X = [x[y!=0] for y in Y]
        Y = [y[y!=0] for y in Y]
        return X,Y
    return [x]*len(Z),Y
#%%
def calculate_L_wonx(neiz,n=None):
    if n is None:
        n = len(neiz)
    L = np.zeros(n,int)
    for i in np.arange(n):
        a = set(neiz[i])
        if len(a) == 0:
            continue
        for j in a:
            L[i] += len(a.intersection(neiz[j]))
    return L
def calculate_L_sub_wonx(neiz,sub):
    L = []
    for i in sub:
        a = set(neiz[i])
        L.append(0)
        if len(a) == 0:
            continue
        L[-1] = sum([len((a.intersection(neiz[j]))) for j in a])
        # for j in a:
            # L[-1] += len((a.intersection(neiz[j])))
    return L
def cluster_coefs(degree,neiz,return_only_mean=False,n=None):
    if n is None:
        n = len(degree)
    wh = degree>1
    c = calculate_L_wonx(neiz,n)
    if return_only_mean:
        return (c/(degree*(degree-1))).mean() if wh.all() else (c[wh]/(degree[wh]*(degree[wh]-1))).mean()
    else:
        return c/(degree*(degree-1)) if wh.all() else c[wh]/(degree[wh]*(degree[wh]-1))
def cluster_coefs_sub(degree,neiz,sub):
    c = calculate_L_sub_wonx(neiz,sub)
    wh = degree > 1
    if wh.all():
        return c/(degree*(degree-1))
    else:
        c[wh] = c[wh]/(degree[wh]*(degree[wh]-1))
        return c
#%%
def complete_graph_neizonly(n,number_of_empty_nodes=0):
    if n<=0:
        print(f"ERROR: n must be positive number in "+complete_graph_neizonly.__name__)
        return []
    if number_of_empty_nodes>0:
        nrange = list(np.arange(n))
        return [nrange[:i]+nrange[i+1:] for i in range(n-number_of_empty_nodes)]+[[] for i in range(number_of_empty_nodes)]
    else:
        nrange = list(np.arange(n))
        return [nrange[:i]+nrange[i+1:] for i in range(n)]
def complete_graph(n,number_of_empty_nodes=0,return_degree=True,return_L=True,return_clustcoef=False,return_adj=False):
    if n<=0:
        print(f"ERROR: n must be positive number in "+complete_graph.__name__)
        return []
    r = [complete_graph_neizonly(n-number_of_empty_nodes,number_of_empty_nodes)]
    if number_of_empty_nodes>0:
        m = n-number_of_empty_nodes
        if return_degree:
            r.append(np.zeros(n,int))
            r[-1][:m] = m-1
        if return_L:
            r.append(np.zeros(n,int))
            r[-1][:m] = m*(m-1)
        if return_clustcoef:
            r.append(np.zeros(n))
            r[-1][:m] = 1
        if return_adj:
            r.append(np.zeros((n,n),bool))
            r[-1][:m,:m] = True
            r[-1][:m,:m][np.diag_indices(m)] = False
    else:
        if return_degree:
            r.append(np.full(n,n-1,int))
        if return_L:
            r.append(np.full(n,n*(n-1),int))
        if return_clustcoef:
            r.append(np.full(n,1))
        if return_adj:
            r.append(np.ones((n,n),bool))
            r[-1][np.diag_indices(n)] = False
    if len(r) == 1:
        return r[0]
    else:
        return tuple(r)
#%%
