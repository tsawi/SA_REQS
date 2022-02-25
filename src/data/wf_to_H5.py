#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:00:19 2021
example: Parkfield repeaters::
@author: theresasawi
"""

import h5py
import os
import sys
sys.path.append('/Users/theresasawi/Documents/12_Projects/specufex_processing/1_preprocessing/functions/')
import generators


def wf_to_H5(pathProj,dataH5_path,wf_filelist,lenData,channel_ID,station,channel):
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
    wf_filelist : list of str
        Absolute paths to wavefiles.
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
        
        
    try:
        del gen_wf
    except:
        pass
        
        
    filenames = list(cat["filename"])
    
##    # define generator (function)
    # gen_wf = generators.gen_wf_from_folder(filenames,
    #                             lenData,
    #                             channel_ID)    
    
    
    
    
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
            data = generators.load_wf(pathWF+ev["filename"], lenData, channel_ID)
            if data is not None:
                channel_group.create_dataset(name=str(ev["ev_ID"]), data=data)
                evID_keep.append(ev["ev_ID"])
            else:
                print(ev.ev_ID, " not saved")
    
 
    print(dupl_evID, ' duplicate events found and avoided')
    print(len(evID_keep), ' waveforms loaded')
    
    
    
    return evID_keep
    
    #%%
    
    
    #%%
