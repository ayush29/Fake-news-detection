# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 19:30:02 2018

@author: Ayush
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def plotClaimSources(fname):
    sources = pd.read_csv(fname,names=['source','f1','f2','f3','f4','f5','f6','f7'])
    x = sources.iloc[:,1:8].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    df = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
    df = pd.concat([sources.iloc[:,0:1],df],axis = 1)
    fig, ax = plt.subplots()
    ax.scatter(df.loc[:5,'pc1'], df.loc[:5,'pc2'])
    for i, txt in enumerate(df.loc[:5,'source']):
        ax.annotate(txt, (df.loc[i,'pc1'], df.loc[i,'pc2']))
    


def plotArticleSources(fname):
    sources = pd.read_csv(fname,names=['source','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12'])
    x = sources.iloc[:,1:13].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    df = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
    df = pd.concat([sources.iloc[:,0:1],df],axis = 1)
    fig, ax = plt.subplots()
    ax.scatter(df.loc[:5,'pc1'], df.loc[:5,'pc2'])
    for i, txt in enumerate(df.loc[:5,'source']):
        ax.annotate(txt, (df.loc[i,'pc1'], df.loc[i,'pc2']))
        
plotArticleSources('./members.csv')