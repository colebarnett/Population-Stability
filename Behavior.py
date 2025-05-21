# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:24:39 2025

@author: coleb
"""
import tables
import os
# import sys
# import time
# import warnings
import matplotlib.pyplot as plt
import numpy as np
# import tdt
# import pandas as pd
# from matplotlib.patches import Ellipse
import scipy as sp



def get_HDFstate_TDT_LFPsamples(ind_state,state_time,syncHDF_file):
    '''
    This method finds the TDT sample numbers that correspond to indicated task state using the syncHDF.mat file.

    Inputs:
        - ind_state: array with state numbers corresponding to which state we're interested in finding TDT sample numbers for, e.g. self.ind_hold_center_states
        - state_time: array of state times taken from corresponding hdf file
        - syncHDF_file: syncHDF.mat file path, e.g. '/home/srsummerson/storage/syncHDF/Mario20161104_b1_syncHDF.mat'
    Output:
        - lfp_state_row_ind: array of tdt sample numbers that correspond the the task state events in ind_state array
    '''
    # Load syncing data
    hdf_times = dict()
    sp.io.loadmat(syncHDF_file, hdf_times)
    #print(syncHDF_file)
    hdf_rows = np.ravel(hdf_times['row_number'])
    hdf_rows = [val for val in hdf_rows]
    #print(hdf_times['tdt_dio_samplerate'])
    dio_tdt_sample = np.ravel(hdf_times['ripple_samplenumber'])
    dio_freq = np.ravel(hdf_times['ripple_dio_samplerate'])

    lfp_dio_sample_num = dio_tdt_sample  # assumes DIOx and LFPx are saved using the same sampling rate

    state_row_ind = state_time[ind_state]        # gives the hdf row number sampled at 60 Hz
    lfp_state_row_ind = np.zeros(state_row_ind.size)

    for i in range(len(state_row_ind)):
        hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind[i])) #find index of row we are on
        if np.abs(hdf_rows[hdf_index] - state_row_ind[i])==0: #if time of state matches hdf row timestamp
            lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
        elif hdf_rows[hdf_index] > state_row_ind[i]: #if times don't match up, do linear interp with the prev timestamp
            hdf_row_diff = hdf_rows[hdf_index] - hdf_rows[hdf_index -1]  # distance of the interval of the two closest hdf_row_numbers
            m = (lfp_dio_sample_num[hdf_index]-lfp_dio_sample_num[hdf_index - 1])/hdf_row_diff
            b = lfp_dio_sample_num[hdf_index-1] - m*hdf_rows[hdf_index-1]
            lfp_state_row_ind[i] = int(m*state_row_ind[i] + b)
        elif (hdf_rows[hdf_index] < state_row_ind[i])&(hdf_index + 1 < len(hdf_rows)): #if times don't match up, do linear interp with the next timestamp
            hdf_row_diff = hdf_rows[hdf_index + 1] - hdf_rows[hdf_index]
            if (hdf_row_diff > 0):
                m = (lfp_dio_sample_num[hdf_index + 1] - lfp_dio_sample_num[hdf_index])/hdf_row_diff
                b = lfp_dio_sample_num[hdf_index] - m*hdf_rows[hdf_index]
                lfp_state_row_ind[i] = int(m*state_row_ind[i] + b)
            else:
                lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
        else:
            lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]

    return lfp_state_row_ind, dio_freq
    
    
class CenterOut():
    '''

    '''

    def __init__(self, hdf_files): #, num_trials_A, num_trials_B):
        for i, hdf_file in enumerate(hdf_files): 
            self.filename =  hdf_file
            table = tables.open_file(self.filename)
            if i == 0:
                self.state = table.root.task_msgs[:]['msg']
                self.state_time = table.root.task_msgs[:]['time']

            else:
                self.state = np.append(self.state, table.root.task_msgs[:]['msg'])
                self.state_time = np.append(self.state_time, self.state_time[-1] + table.root.task_msgs[:]['time'])
        
        
        self.ind_reward_states = np.ravel(np.nonzero(self.state == b'reward')) #Successful trials
        self.ind_target_prompt = self.ind_reward_states - 3 #start of mvmt
        self.ind_target_hold = self.ind_reward_states - 2 #end of mvmt
        
        self.fs_hdf = 60 #Hz
        
        self.block_type = table.root.task[:]['block_type'][:,0]
        
        
        ## Get target locations (from Hannah's code)
        fix = lambda r:r+(2*np.pi) if r < 0 else r #lambda d:d+360 if d < 0 else d
        
        target_xloc = table.root.task[:]['target'][:,0]
        target_yloc = table.root.task[:]['target'][:,2]
        target_xloc = target_xloc[self.state_time[self.ind_target_prompt]]
        target_yloc = target_yloc[self.state_time[self.ind_target_prompt]]
        target_loc_rads = np.array([fix(np.arctan2(y,x)) for y,x in zip(target_xloc, target_yloc)])
        self.target_loc_degs = (np.array(target_loc_rads)*(180/np.pi)).astype(int) # for each trial
