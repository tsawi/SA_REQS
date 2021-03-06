#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 05:34:57 2021

@author: theresasawi
"""
import scipy as sp


import h5py
import numpy as np
import sys
import obspy
import os

sys.path.append('functions/')

import tables
tables.file._open_files.close_all()
import scipy.io as spio
import scipy.signal


#%%



def getEventID(path,key,eventID_string):
    """
    Generate unique event ID based on filename


    """
    evID = eval(eventID_string)
    # print(evID)

    # if 'Parkfield' in key:
    #
    #     evID = path.split('/')[-1].split('.')[-1]
    #
    # else:## default:: event ID is the waveform filename
    #
    #     evID  = path.split('/')[-1].split('.')[0]


    return evID


#%%

def load_wf(filename, lenData, channel_ID=None):
    """Loads a waveform file and returns the data.

    Arguments
    ---------
    filename: str
        Filename to load
    lenData: int
        Number of samples in the file. Must be uniform for all files.
    channel_ID: int
        If the fileis an obspy stream, this is the desired channel.
    """
    if ".txt" in filename:
        data = np.loadtxt(filename)
    else: #catch loading errors
        st = obspy.read(filename)
        ### REMOVE RESPONSE ??
        st.detrend('demean')
        data = st[channel_ID].data

    #make sure data same length
    Nkept = 0
    if len(data)==lenData:
        return data
    #Parkfield is sometimes one datapoint off
    elif np.abs(len(data) - lenData) ==1:
        data = data[:-1]
        Nkept += 1
        return data

    else:
        print(filename, ': data wrong length')
        print(f"this event: {len(data)}, not {lenData}")
        return None

def gen_wf_from_folder(wf_filelist,lenData,channel_ID):
    """
    Note
    ----------
   ** MAKE NEW FOR EACH DATASET:: Add settings for your project key below

    Parameters
    ----------
    wf_filelist : list of paths to waveforms
    lenData : number of samples in data (must be same for all data)
    channel_ID : for obspy streams, this is the index of the desired channel

    Yields
    ------
    evID : formatted event ID
    Nkept : number of kept wfs

    """

    Nkept=0 # count number of files kept
    Nerr = 0 # count file loading error
    NwrongLen = 0

    for i, path in enumerate(wf_filelist):
        if ".txt" in path:
            data = np.loadtxt(path)
        else: #catch loading errors
            st = obspy.read(path)
            ### REMOVE RESPONSE ??
            st.detrend('demean')
            data = st[channel_ID].data

        #make sure data same length
        if len(data)==lenData:
            Nkept += 1
            if i%100==0:
                print(f"{i}/{len(wf_filelist)}")
            yield data, Nkept

        #Parkfield is sometimes one datapoint off
        elif np.abs(len(data) - lenData) ==1:
            data = data[:-1]
            Nkept += 1
            yield data, Nkept

        else:
            NwrongLen += 1
            print(NwrongLen, ' data wrong length')
            print(f"this event: {len(data)}, not {lenData}")


# make sgram generator
def gen_sgram_QC(key,evID_list,dataH5_path,trim=True,saveMat=False,sgramOutfile='.',**args):

    fs=args['fs']
    nperseg=args['nperseg']
    noverlap=args['noverlap']
    nfft=args['nfft']
    mode=args['mode']
    scaling=args['scaling']
    fmin=args['fmin']
    fmax=args['fmax']
    Nkept = 0
    evID_BADones = []
    for i, evID in enumerate(evID_list):

        if i%100==0:
            print(i,'/',len(evID_list))

        with h5py.File(dataH5_path,'a') as fileLoad:
            stations=args['station']
            data = fileLoad[f"waveforms/{stations}/{args['channel']}"].get(str(evID))[:]


        fSTFT, tSTFT, STFT_raw = sp.signal.spectrogram(x=data,
                                                    fs=fs,
                                                    nperseg=nperseg,
                                                    noverlap=noverlap,
                                                    #nfft=Length of the FFT used, if a zero padded FFT is desired
                                                    nfft=nfft,
                                                    scaling=scaling,
                                                    axis=-1,
                                                    mode=mode)

        if trim:
            freq_slice = np.where((fSTFT >= fmin) & (fSTFT <= fmax))
            #  keep only frequencies within range
            fSTFT   = fSTFT[freq_slice]
            STFT_0 = STFT_raw[freq_slice,:][0]
        else:
            STFT_0 = STFT_raw
            # print(type(STFT_0))


        # =====  [BH added this, 10-31-2020]:
        # Quality control:
        if np.isnan(STFT_0).any()==1 or  np.median(STFT_0)==0 :
            if np.isnan(STFT_0).any()==1:
                print('OHHHH we got a NAN here!')
                #evID_list.remove(evID_list[i])
                evID_BADones.append(evID)
                pass
            if np.median(STFT_0)==0:
                print('OHHHH we got a ZERO median here!!')
                #evID_list.remove(evID_list[i])
                evID_BADones.append(evID)
                pass

        if np.isnan(STFT_0).any()==0 and  np.median(STFT_0)>0 :

            normConstant = np.median(STFT_0)

            STFT_norm = STFT_0 / normConstant  ##norm by median

            STFT_dB = 20*np.log10(STFT_norm, where=STFT_norm != 0)  ##convert to dB
            # STFT_shift = STFT_dB + np.abs(STFT_dB.min())  ##shift to be above 0
    #

            STFT = np.maximum(0, STFT_dB) #make sure nonnegative


            if  np.isnan(STFT).any()==1:
                print('OHHHH we got a NAN in the dB part!')
                evID_BADones.append(evID)
                pass
            # =================save .mat file==========================

            else:

                Nkept +=1

                if saveMat==True:
                    if not os.path.isdir(sgramOutfile):
                        os.mkdir(sgramOutfile)


                    spio.savemat(sgramOutfile + str(evID) + '.mat',
                              {'STFT':STFT,
                                'fs':fs,
                                'nfft':nfft,
                                'nperseg':nperseg,
                                'noverlap':noverlap,
                                'fSTFT':fSTFT,
                                'tSTFT':tSTFT})


            # print(type(STFT))

            yield evID,STFT,fSTFT,tSTFT, normConstant, Nkept,evID_BADones, i





## simpler way to load waveforms into H5

def wf_to_H5(pathProj,dataH5_path,pathWF,cat,lenData,channel_ID,station,channel):
    """
    Load waveforms and store as arrays in H5 file
    (Does not store catalog -- that should be a different function)
    Returns the event IDs of successfully loaded waveforms

    Parameters
    ----------
    pathProj : str
        .
    dataH5_path : str
        .
     cat : pandas Dataframe
        Has columns called "ev_ID" and "filename".
    lenData : int
        In samples.
    channel_ID : int
        For obspy stream object.
    station : str
        .
    channel : str
        ex. N, EHZ.

    Returns
    -------
    evID_keep : list of str
        Ev_IDs of successfully loaded waveforms.

    """
   
###     clear old H5 if it exists, or else error will appear
    if os.path.exists(dataH5_path):
        os.remove(dataH5_path)
        

        
    
    evID_keep = []
    



    with h5py.File(dataH5_path,'a') as h5file:
    
    
        h5file.create_group("waveforms")
        h5file.create_group(f"waveforms/{station}")
        channel_group = h5file.create_group(f"waveforms/{station}/{channel}")
    
    
    
        dupl_evID = 0
        n=0
    
        for n, ev in cat.iterrows():
            if n%500==0:
                print(n, '/', len(cat))
            data = load_wf(pathWF+ev["filename"], lenData, channel_ID)
            if data is not None:
                channel_group.create_dataset(name=str(ev["ev_ID"]), data=data)
                evID_keep.append(ev["ev_ID"])
            else:
                print(ev.ev_ID, " not saved")
    
 
    print(dupl_evID, ' duplicate events found and avoided')
    print(len(evID_keep), ' waveforms loaded')
    
    
    
    return evID_keep
    