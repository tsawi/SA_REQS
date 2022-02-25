#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 09:15:16 2021

For SAREQ; functions pulled from Gorner as needed

@author: theresasawi
"""


import h5py
import pandas as pd
import numpy as np
#from obspy import read
from matplotlib import pyplot as plt

import datetime as dtt

import datetime
from scipy.stats import kurtosis
from  sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import MinMaxScaler
from scipy import spatial

from scipy.signal import butter, lfilter
#import librosa
# # sys.path.insert(0, '../01_DataPrep')
from scipy.io import loadmat
from sklearn.decomposition import PCA
# sys.path.append('.')
from sklearn.metrics import silhouette_samples
import scipy as sp
import scipy.signal


from obspy.signal.cross_correlation import correlate, xcorr_max

from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import sklearn.metrics

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def linearizeFP(SpecUFEx_H5_path,ev_IDs):
    """
    Linearize fingerprints, stack into array 

    Parameters
    ----------
    SpecUFEx_H5_path :str
    cat00 : pandas dataframe

    Returns
    -------
    X : numpy array
        (Nevents,Ndim)

    """

    X = []
    with h5py.File(SpecUFEx_H5_path,'r') as MLout:
        for evID in ev_IDs:
            fp = MLout['fingerprints'].get(str(evID))[:]
            linFP = fp.reshape(1,len(fp)**2)[:][0]
            X.append(linFP)

    X = np.array(X)

    return X




# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def PVEofPCA(X,numPCMax=100,cum_pve_thresh=.86,stand='MinMax',verbose=1):
    """
    Calculate cumulative percent variance explained for each principal component.

    Parameters
    ----------
    X : numpy array
        Linearized fingerprints.
    numPCMax : int, optional
        Maximum number of principal components. The default is 100.
    cum_pve_thresh : int or float, optional
        Keep PCs until cumulative PVE reaches threshold. The default is .8.
    stand : str, optional
        Parameter for SKLearn's StandardScalar(). The default is 'MinMax'.

    Returns
    -------
    PCA_df : pandas dataframe
        Columns are PCs, rows are event.
    numPCA : int
        Number of PCs calculated.
    cum_pve : float
        Cumulative PVE.

    """


    if stand=='StandardScaler':
        X_st = StandardScaler().fit_transform(X)
    elif stand=='MinMax':
        X_st = MinMaxScaler().fit_transform(X)
    else:
        X_st = X


    numPCA_range = range(1,numPCMax)


    for numPCA in numPCA_range:

        sklearn_pca = PCA(n_components=numPCA)

        Y_pca = sklearn_pca.fit_transform(X_st)

        pve = sklearn_pca.explained_variance_ratio_

        cum_pve = pve.sum()
        
        if verbose:
            print(numPCA,cum_pve)
        
        if cum_pve >= cum_pve_thresh:
            

            print('\n break \n')
            break


    print('numPCA', numPCA,'; cum_pve',cum_pve)

    pc_cols = [f'PC{pp}' for pp in range(1,numPCA+1)]

    PCA_df = pd.DataFrame(data = Y_pca, columns = pc_cols)


    return PCA_df, numPCA, cum_pve


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def findKmeansKopt(X,range_n_clusters,clusterMetric='silhScore',verbose=1):
    """
    Calculate kmeans, find optimal k number clusters.


    Parameters
    ----------
    X : numpy array
        MxN matrix of M instances and N features.
    range_n_clusters : range type
        Range of K for kmeans; minimum 2.
    clusterMetric : str, optional
        under construction: 'eucDist'.  The default is 'silhScore'.

    Returns
    -------
    Kopt : int 
        Optimal number of clusters.
    cluster_labels_best : list 
        Cluster labels.

    """

    

    metric_thresh = 0
    elbo_plot = []


    for n_clusters in range_n_clusters:

        print(f"kmeans on {n_clusters} clusters...")

        ### SciKit-Learn's Kmeans
        kmeans = KMeans(n_clusters=n_clusters,
                           max_iter = 500,
                           init='k-means++', #how to choose init. centroid
                           n_init=10, #number of Kmeans runs
                           random_state=0) #set rand state

        
        # #kmeans loss function
        # elbo_plot.append(kmeans.inertia_)        
        
        
        ####  Assign cluster labels
        #increment labels by one to match John's old kmeans code
        cluster_labels = [int(ccl)+1 for ccl in kmeans.fit_predict(X)]

        
        
        if clusterMetric == 'silhScore':
            # Compute the silhouette scores for each sample
            silh_scores = silhouette_samples(X, cluster_labels)
            silh_scores_mean = np.mean(silh_scores)
            
            if verbose:
                print('max silh score:',np.max(silh_scores))
                print('min silh score:',np.min(silh_scores))            
                print('mean silh score:',silh_scores_mean)            
                
            
            if silh_scores_mean > metric_thresh:
                Kopt = n_clusters
                metric_thresh = silh_scores_mean
                cluster_labels_best = cluster_labels
                print('max mean silhouette score: ', silh_scores_mean)

            
#         elif clusterMetric == 'eucDist':
                # ... see selectKmeans()

    print(f"Best cluster: {Kopt}")


    return Kopt, cluster_labels_best, metric_thresh, elbo_plot
    
    
    
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
    
    

def selectKmeans(X,ev_IDs,Kopt,clusterMetric='silhScore'):
    """
    Calculate kmeans for select number of clusters.


    Parameters
    ----------
    X : numpy array
        MxN matrix of M instances and N features.
    ev_IDs : list
        List of event IDs for rows of X (to make dataframe for merging)        
    Kopt : int
        Number of clusters for kmeans
    clusterMetric : str, optional
        under construction: 'eucDist'.  The default is 'silhScore'.

    Returns
    -------
    Kopt_df : pandas dataframe
        columns are "ev_ID", "Cluster", "SS", "euc_dist"

    """

    

    

    n_clusters = Kopt


    ### SciKit-Learn's Kmeans
    kmeans = KMeans(n_clusters=n_clusters,
                       max_iter = 500,
                       init='k-means++', #how to choose init. centroid
                       n_init=10, #number of Kmeans runs
                       random_state=0) #set rand state

    
    
    ####  Assign cluster labels
    cluster_labels_0 = kmeans.fit_predict(X)
    #increment labels by one to match John's old kmeans code
    cluster_labels = [int(ccl)+1 for ccl in cluster_labels_0]

    
    
    # Compute the silhouette scores for each sample
    silh_scores = silhouette_samples(X, cluster_labels)
    silh_scores_mean = np.mean(silh_scores)
    print('max mean silhouette score: ', silh_scores_mean)



    #get euclid dist to centroid for each point
    sqr_dist = kmeans.transform(X)**2 #transform X to cluster-distance space.
    sum_sqr_dist = sqr_dist.sum(axis=1)
    euc_dist = np.sqrt(sum_sqr_dist)
            
    # #save centroids
    # centers = kmeans.cluster_centers_ 
    # #kmeans loss function
    # sse = kmeans.inertia_

    
    Kopt_df = pd.DataFrame(
              {'ev_ID':ev_IDs,
               'Cluster':cluster_labels,
               'SS':silh_scores,
               'euc_dist':euc_dist
               })




    return Kopt_df, silh_scores_mean


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def getTopFCat(cat0,topF,startInd=0,distMeasure = "SilhScore"):
    """


    Parameters
    ----------
    cat00 : all events
    topf : get top F events in each cluster
    startInd : can skip first event if needed
    Kopt : number of clusters
    distMeasure : type of distance emtrix between events. Default is "SilhScore",
    can also choose euclidean distance "EucDist"

    Returns
    -------
    catall : TYPE
        DESCRIPTION.

    """


    cat0['event_ID'] = [int(f) for f in  cat0['event_ID']]
    if distMeasure == "SilhScore":
        cat0 = cat0.sort_values(by='SS',ascending=False)

    if distMeasure == "EucDist":
        cat0 = cat0.sort_values(by='euc_dist',ascending=True)

    # try:
    cat0 = cat0[startInd:startInd+topF]
    # except: #if less than topF number events in cluster
    #     print(f"sampled all {len(cat0)} events in cluster!")

    # overwriting cat0 ?????
    return cat0




# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def plotPCA(cat00,catall,Kopt, colors,size=5,size2=15, alpha=.5,labelpad = 5,fontsize=8,ax=None, fig=None):


    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()



    for k in range(1,Kopt+1):

        catk = cat00[cat00.Cluster == k]
        ax.scatter(catk.PC1,catk.PC3,catk.PC2,
                      s=size,
                      marker='x',
                      color=colors[k-1],
                      alpha=alpha)


# plot top FPs
    ax.scatter(catall.PC1,catall.PC3,catall.PC2,
                      s=size2,
                      marker='x',
                      color='k',
                      alpha=1)

    # sm = plt.cm.ScalarMappable(cmap=colors,
    #                            norm=plt.Normalize(vmin=df_stat.Cluster(),vmax=df_stat.Cluster()))
    axLabel = 'PC'#'Principal component '#label for plotting
    # cbar = plt.colorbar(sm,label=stat_name,shrink=.6,pad=.3);

    ax.set_xlabel(f'{axLabel} 1',labelpad=labelpad, fontsize = fontsize);
    ax.set_ylabel(f'{axLabel} 3',labelpad=labelpad, fontsize = fontsize);
    ax.set_zlabel(f'{axLabel} 2',labelpad=labelpad, fontsize = fontsize);
#     plt.colorbar(ticks=range(6), label='digit value')
#     plt.clim(-0.5, 5.5)

    ### Tick formatting is currently hard-coded
    # ax.set_xlim(-.6,.6)
    # ax.set_ylim(-.6,.6)
    # ax.set_zlim(-.6,.6)

    # ticks =  np.linspace(-.6,.6,5)
    # tick_labels = [f'{t:.1f}' for t in ticks]
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(tick_labels)
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(tick_labels)
    # ax.set_zticks(ticks)
    # ax.set_zticklabels(tick_labels)


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo



####FILTERs
##########################################################################################




def butter_bandpass(fmin, fmax, fs, order=5):
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, fmin, fmax, fs, order=5):
    b, a = butter_bandpass(fmin, fmax, fs, order=order)
    y = lfilter(b, a, data)
    return y


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def getWF(evID,dataH5_path,station,channel,fmin,fmax,fs):
    """
    Load waveform data from H5 file, apply 4th order bandpass filter,
    and zero mean

    Parameters
    ----------
    evID : int
    .
    dataH5_path : str
    .
    station : str
    .
    channel : str
    .
    fmin : int or float
    Minimum frequency for bandpass filter.
    fmax : int or float
    Maximum frequency for bandpass filter.
    fs : int
    Sampling rate.

    Returns
    -------
    wf_zeromean : numpy array
    Filtered and zero-meaned waveform array.

    """

    with h5py.File(dataH5_path,'a') as fileLoad:

        wf_data = fileLoad[f'waveforms/{station}/{channel}'].get(str(evID))[:]

    wf_filter = butter_bandpass_filter(wf_data, fmin,fmax,fs,order=4)
    wf_zeromean = wf_filter - np.mean(wf_filter)

    return wf_zeromean

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo



def calcCCMatrix(catRep,shift_cc,dataH5_path,station,channel,fmin,fmax,fs):
    '''
    catRep   : (pandas.Dataframe) catalog with event IDs

    shift_cc : (int) Number of samples to shift for cross correlation.
                    The cross-correlation will consist of 2*shift+1 or
                    2*shift samples. The sample with zero shift will be in the middle.


    Returns np.array

    '''

    cc_mat = np.zeros([len(catRep),len(catRep)])
    lag_mat = np.zeros([len(catRep),len(catRep)])

    for i in range(len(catRep)):
        for j in range(len(catRep)):

            evIDA = catRep.event_ID.iloc[i]
            evIDB = catRep.event_ID.iloc[j]


            wf_A = getWF(evIDA,dataH5_path,station,channel,fmin=fmin,fmax=fmax,fs=fs)
            wf_B = getWF(evIDB,dataH5_path,station,channel,fmin=fmin,fmax=fmax,fs=fs)



            cc = correlate(wf_A, wf_B, shift_cc)
            lag, max_cc = xcorr_max(cc)

            cc_mat[i,j] = max_cc
            lag_mat[i,j] = lag


    return cc_mat,lag_mat

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def calcCorr_template(wf_A,catRep,shift_cc,dataH5_path,station,channel,fmin,fmax,fs):
    ''' Calculate cross-correlation matrix and lag time for max CC coeg=f

    wf_A     : (np.array) wf template to match other waveforms

    catRep   : (pandas.Dataframe) catalog with event IDs

    shift_cc : (int) Number of samples to shift for cross correlation.
                    The cross-correlation will consist of 2*shift+1 or
                    2*shift samples. The sample with zero shift will be in the middle.


    Returns np.array
    '''


    cc_vec = np.zeros([len(catRep)]) #list cc coef

    lag_vec = np.zeros([len(catRep)]) #list lag time (samples) to get max cc coef


    for j in range(len(catRep)):

        evIDB = catRep.event_ID.iloc[j]

        wf_B = getWF(evIDB,dataH5_path,station,channel,fmin=fmin,fmax=fmax,fs=fs)



        cc = correlate(wf_A, wf_B, shift_cc)
        lag, max_cc = xcorr_max(cc)

        cc_vec[j] = max_cc
        lag_vec[j] = lag


    return cc_vec,lag_vec

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def lagWF_Scalar(waveform, lag0, index_wf):
    """
    Shift waveform amount to match template waveform.


    Parameters
    ----------
    waveform : np.array

    lag0 : int
        Scalar from lag vector lag_vec, output of calcCorr_template.
    index_wf : int
        Index of waveform in catRep catalog.

    Returns
    -------
     np.array - time-shifted waveform to get maxCC

    """


    i = index_wf

    #
    if lag0<0:
#         print('neg lag', lag0[i])
        isNeg = 1
        lag00 = int(np.abs(lag0)) #convert to int
    else:
#         print('pos lag', lag0[i])
        isNeg = 0
        lag00 = int(lag0)


    padZ = np.zeros(lag00)# in samples
    padZ = np.ones(lag00)*np.nan# in samples

    if isNeg:
        waveform_shift = np.hstack([waveform,padZ])
        waveform_shift2 = waveform_shift[lag00:]

    else:
        waveform_shift = np.hstack([padZ,waveform])
        waveform_shift2 = waveform_shift[:-lag00]

    if lag0[i]==0 or lag0[i]==0.0:
        waveform_shift2 = waveform

    return waveform_shift2


def lagWF(waveform, lag0, index_wf):
    """
    Shift waveform amount to match template waveform.


    Parameters
    ----------
    waveform : np.array

    lag0 : np.array
        Matrix of lag times; output of calcCC_Mat.
    index_wf : int
        Index of waveform in catalog.

    Returns
    -------
     np.array - time-shifted waveform to get maxCC

    """


    i = index_wf

    #
    if lag0[i]<0:
#         print('neg lag', lag0[i])
        isNeg = 1
        lag00 = int(np.abs(lag0[i])) #convert to int
    else:
#         print('pos lag', lag0[i])
        isNeg = 0
        lag00 = int(lag0[i])


    padZ = np.zeros(lag00)# in samples
    padZ = np.ones(lag00)*np.nan# in samples

    if isNeg:
        waveform_shift = np.hstack([waveform,padZ])
        waveform_shift2 = waveform_shift[lag00:]

    else:
        waveform_shift = np.hstack([padZ,waveform])
        waveform_shift2 = waveform_shift[:-lag00]

    if lag0[i]==0 or lag0[i]==0.0:
        waveform_shift2 = waveform

    return waveform_shift2


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOoOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo






def swapLabels(cat,A,B):
    """
    Swap labels bewteen Cluster A and Cluster B.

    Parameters
    ----------
    cat : pd.DataFrame
        Must have column names: event_ID, Cluster.
    A : int, str
        Original cluster assignment.
    B : int, str
        New cluster assignment.

    Returns
    -------
    pd.DataFrame

    """



## swap label A to B
    dummy_variable = 999
    cat_swapped = cat.copy()
    cat_swapped.Cluster = cat_swapped.Cluster.replace(A,dummy_variable)
    cat_swapped.Cluster = cat_swapped.Cluster.replace(B,A)
    cat_swapped.Cluster = cat_swapped.Cluster.replace(dummy_variable,B)


    return cat_swapped




# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOoOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo




def catMergeFromH5(path_Cat,path_proj,outfile_name):
    '''
    Keep csv catalog events based on H5 used in SpecUFEx

    '''

    ## read 'raw' catalog, the immutable one
    cat_raw = pd.read_csv(path_Cat)
    cat_raw['event_ID'] = [str(int(evv)) for evv in cat_raw['event_ID']]


    ## load event IDs from H5
    MLout =  h5py.File(path_proj + outfile_name,'r')
    evID_kept = [evID.decode('utf-8') for evID in MLout['catalog/event_ID/'][:]]
    MLout.close()

    ## put H5 events into pandas dataframe
    df_kept = pd.DataFrame({'event_ID':evID_kept})

    ## merge based on event ID
    cat00 = pd.merge(cat_raw,df_kept,on='event_ID')

    ## if length of H5 events and merged catalog are equal, then success
    if len(evID_kept) == len(cat00):
        print(f'{len(cat00)} events kept, merge sucessful')
    else:
        print('check merge -- error may have occurred ')


    ## convert to datetime, set as index
    cat00['datetime'] = [pd.to_datetime(i) for i in cat00.datetime]
    cat00['datetime_index']= [pd.to_datetime(i) for i in cat00.datetime]
    cat00 = cat00.set_index('datetime_index')


    return cat00


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo




def plotMapGMT(cat_feat,stat_df,buff,colorBy='depth_km',maxColor=10):


    cat_star = cat_feat[colorBy]

    region = [
        cat_feat.long.min() - buff,
        cat_feat.long.max() + buff,
        cat_feat.lat.min() - buff,
        cat_feat.lat.max() + buff,
    ]

    print(region)


    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M15c", frame=True)
    fig.coast(land="black", water="skyblue")

    if maxColor is not None:
        pygmt.makecpt(cmap="viridis", series=[cat_star.min(), maxColor])
    else:
        pygmt.makecpt(cmap="viridis", series=[cat_star.min(), cat_star.max()])

    fig.plot(
        x=cat_feat.long,
        y=cat_feat.lat,
        size=0.05 * 2 ** cat_feat.magnitude,
        color=cat_star,
        cmap=True,
        style="cc",
        pen="black",
    )

    if 'depth' in colorBy:
        fig.colorbar(frame='af+l"Depth (km)"')

    else:
        fig.colorbar(frame=f'af+l"{colorBy}"')

    fig.plot(x=stat_df.long, y=stat_df.lat,style="t.5c", color="pink", pen="black")




    fig.show()

    return fig




# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo