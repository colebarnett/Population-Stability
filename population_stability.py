# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:35:03 2025

@author: coleb
"""
#### Define global things

import os
import sys
import time
import pickle
import itertools

import numpy as np
import pandas as pd
from scipy import signal
# import statsmodels.api as sm
from matplotlib import pyplot as plt
# from matplotlib_venn import venn2,venn3
from mne_connectivity import spectral_connectivity_epochs

import Behavior




TEST = False
CHS = np.arange(0,128,step=8,dtype=int) #define subset of channels to use
SNIPPET_DURATION = 0.9 #sec. Trials shorter than this will be discarded, trials longer this will be truncated.


FREQ_BANDS = {'theta':(3.5,8.5), # Define freq bands (Hz)
              'alpha':(8.5,12.5),
              'beta':(12.5,34.5),
              'gamma':(34.5,200.5),
              'low gamma':(34.5,60.5),
              'high gamma':(60.5,200.5)}

NUM_TRIALS_PER_BLOCK = 384 # 48 trial sets per block * 8 directions per trial set
BL_TRIALS = slice(0,NUM_TRIALS_PER_BLOCK)
EP_TRIALS = slice(NUM_TRIALS_PER_BLOCK,NUM_TRIALS_PER_BLOCK*3//2) #only look at the first half of this block
LP_TRIALS = slice(NUM_TRIALS_PER_BLOCK*2,NUM_TRIALS_PER_BLOCK*3)

SESSIONS = ["braz20220315_07_te90",
            "braz20220321_04_te116",
            "braz20220324_19_te176",
            "braz20220401_05_te220",
            "braz20220416_04_te294",
            "braz20220425_04_te320",
            "braz20220504_04_te376",
            "braz20220510_04_te399",
            "braz20220516_04_te425",
            "braz20220611_04_te478",
            "braz20220617_04_te498"]

SMALL_FONTSIZE = 8
MED_FONTSIZE = 12
LARGE_FONTSIZE = 18

# Paths
DATA_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\Population Stability\Data"
BMI_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\bmi_python"
# PROJ_FOLDER = r"F:\cole"
# BMI_FOLDER = r"C:\Users\crb4972\Desktop\bmi_python"


# Import Neuroshare libraries from bmi_python folder
NS_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns')
sys.path.insert(1,BMI_FOLDER) #add bmi_python folder to package search path
sys.path.insert(2, NS_FOLDER) #add neuroshare python folder to package search path
from nsfile import NSFile
    
os.chdir(DATA_FOLDER)




class ProcessSpikes:
    '''
    From .mat, .hdf, and .nev files, load spike times and calculate firing rate 
    aligned to task events.
    
    Output: DataFrame with firing rate for each unit for each trial
    '''
    
    def __init__(self, session: str):

        
        self.session = session
        self.file_prefix = os.path.join(DATA_FOLDER, self.session)
        
#         # [Initiate different data files]
#         self.ns2file = None
#         self.hdffile = None
#         self.matfile = None
#         self.pklfile = None
#         self.ns5file = None
#         self.nevfile = None
        
        ## Data
        self.has_hdf = False # For behavior
        self.has_ns5 = False # For syncing 
        self.has_ns2 = False # For LFP
        self.has_nev = False # For waveform 
        self.has_decoder = False # For decoder
        self.has_mat = False # For syncing
        self.has_pkl = False # For spike times
        self.check_files()
        
        ## Get Spike Times
        self.spike_times = None 
        self.unit_labels = None
        if self.has_pkl:
            self.load_spike_times()
        else:
            self.get_spike_times()
        
        ## Get Times Align
        self.times_align = None
        self.get_times_align()
        
        ## Get Firing Rates
        self.firing_rate_df = None
        self.t_before = 0.2 #how far to look before time_align point [s]
        self.t_after = 0.0 #how far to look after time_align point [s]
        self.get_firing_rate() 
        
        ## Save Out Processed Data
        # self.df = pd.concat([self.behavior_df,self.firing_rate_df],axis='columns') #merge behavior_df and firing_rate_df
        # self.dict_out = {'Name':session, 'df':self.df}
        self.dict_out = {'Session':session, 'df':self.firing_rate_df}
        
        

    def check_files(self):
        """
        For a proper analysis to be done,
        the hdf, nev, mat, and pkl files are essential.
        The ns5 and ns2 are optional.
        """
        
        if os.path.exists(self.file_prefix + '.hdf'):
            self.has_hdf = True
        if os.path.exists(self.file_prefix + '.ns5'):
            self.has_ns5 = True
        if os.path.exists(self.file_prefix + '.ns2'):
            self.has_ns2 = True
        if os.path.exists(self.file_prefix + '.nev'):
            self.has_nev = True
        if os.path.exists(self.file_prefix + '_syncHDF.mat'):
            self.has_mat = True
        if os.path.exists(self.file_prefix + '_nev_output.pkl'):
            self.has_nev_output = True
        if os.path.exists(self.file_prefix + '_KFDecoder.pkl'):
            self.has_decoder = True
        if os.path.exists(self.file_prefix + '_spike_times_dict.pkl'):
            self.has_pkl = True
        
          
        
    def get_spike_times(self):
        
        assert self.has_nev, FileNotFoundError(f'.nev file not found! Session: {self.session}')

        ## Get Spike Times
        print('Loading spike times..')
        self.nevfile = NSFile(self.file_prefix + '.nev')
        spike_entities = [e for e in self.nevfile.get_entities() if e.entity_type==3]
        headers = np.array([s.get_extended_headers() for s in spike_entities]) #get info for each ch
        # [print(i,h[b'NEUEVLBL'].label[:7]) for i,h in enumerate(headers) if b'NEUEVLBL' in h.keys()]
        unit_idxs = np.nonzero([h[b'NEUEVWAV'].number_sorted_units for h in headers])[0] #get ch idxs where there is a sorted 
        unit_idxs = [unit_idx for unit_idx in unit_idxs if b'NEUEVLBL' in headers[unit_idx].keys()] #exclude entities without NEUEVLBL field
        if TEST: unit_idxs=unit_idxs[:3] #to make runtime shorter for testing
        self.num_units = len(unit_idxs)
        self.unit_labels = ["Unit " + h[b'NEUEVLBL'].label[:7].decode() + self.session for h in headers[unit_idxs] ] #get labels of all sorted units
        # num_units_per_idx = [h[b'NEUEVWAV'].number_sorted_units for h in headers[unit_idxs]]
        [print(f'More than one unit found for {h[b"NEUEVLBL"].label[:7]}!') for h in headers[unit_idxs] if h[b'NEUEVWAV'].number_sorted_units > 1]
        recording_duration = self.nevfile.get_file_info().time_span # [sec]


        self.spike_times = [] #each element is a list spike times for a sorted unit
        # spike_waveforms = [] #each element is a list of waveforms for a sorted unit
        for i,unit_idx in enumerate(unit_idxs): #loop thru sorted unit
            unit = spike_entities[unit_idx]
            self.spike_times.append([]) #initiate list of spike times for this unit
            
            for spike_idx in range(unit.item_count):
                self.spike_times[i].append(unit.get_segment_data(spike_idx)[0])
#                 spike_waveforms.append(unit.get_segment_data(spike_idx)[1])

            print(f'{self.unit_labels[i]}: {unit.item_count} spikes. Avg FR: {unit.item_count/recording_duration:.2f} Hz. ({i+1}/{len(unit_idxs)})')
        print('All spike times loaded!')
        
        # Save out spike times so we don't need to load them from nev again
        spike_times_dict = {'unit_labels':self.unit_labels, 'spike_times':self.spike_times}
        with open(self.file_prefix+'_spike_times_dict.pkl','wb') as f:
            pickle.dump(spike_times_dict,f)
            print(self.file_prefix+'_spike_times_dict.pkl saved!')
        
        return


    def load_spike_times(self):
        
        # Load dict of previously saved spike times
        with open(self.file_prefix+'_spike_times_dict.pkl','rb') as f:
            spike_times_dict = pickle.load(f)
            print(self.session+'_spike_times_dict.pkl' + ' loaded!')
            
        self.unit_labels = spike_times_dict['unit_labels']
        self.spike_times = spike_times_dict['spike_times']
        self.num_units = len(self.unit_labels)
            
        #print out FRs for funsies
        for i,unit_spikes in enumerate(self.spike_times): #loop thru sorted unit
            print(f'{self.unit_labels[i]}: {len(unit_spikes)} spikes. Avg FR: {len(unit_spikes)/(unit_spikes[-1]-unit_spikes[0]):.2f} Hz. ({i+1}/{len(self.unit_labels)})')
        
        return
        
    def get_times_align(self):
        '''
        Gets the array of indices (sample numbers) corresponding to the hold_center, 
        target, and check_reward time points of the given session.
        This facilitates time-aligned analyses.
        
        Parameters
        ----------
        hdf_files : list of hdf files for a single session
        syncHDF_files : list of syncHDF_files files which are used to make the alignment between behavior data and spike data
            
        Outputs
        -------
        target_hold_TDT_ind : 1D array containing the TDT indices for the target hold onset times of the given session
        '''
        
        assert self.has_hdf, FileNotFoundError(f'.hdf file not found! Session: {self.session}')
        assert self.has_mat, FileNotFoundError(f'.mat file not found! Session: {self.session}')
        
        self.hdf_file = self.file_prefix + '.hdf'
        self.syncHDF_file = self.file_prefix + '_syncHDF.mat'
        
        fs_hdf = 60 #hdf fs is always 60
        
        # load behavior data
        cb = Behavior.CenterOut([self.hdf_file]) #method needs hdf filenames in a list
        self.num_trials = cb.num_successful_trials
        
        # Find times of successful trials
#         ind_hold_center = cb.ind_check_reward_states - 4 #times corresponding to hold center onset
#         ind_mvmt_period = cb.ind_check_reward_states - 3 #times corresponding to target prompt
#         ind_reward_period = cb.ind_check_reward_states #times corresponding to reward period onset
        ind_target_hold = cb.ind_check_reward_states - 2 # times corresponding to target hold onset
        
        # align spike tdt times with hold center hdf indices using syncHDF files
        target_hold_TDT_ind, DIO_freq = Behavior.get_HDFstate_TDT_LFPsamples(ind_target_hold,cb.state_time,self.syncHDF_file)

        # Ensure that we have a 1 to 1 correspondence btwn indices we put in and indices we got out.
        assert len(target_hold_TDT_ind) == len(ind_target_hold), f'Repeat hold times! Session: {self.session}'
        assert len(target_hold_TDT_ind) == self.num_trials

        self.times_align = target_hold_TDT_ind / DIO_freq
        
        
        # #Plot to check things out
        # tdt_times = target_hold_TDT_ind / DIO_freq / 60  #convert from samples to seocnds to minutes
        # hdf_times = (cb.state_time[ind_target_hold]) / fs_hdf / 60  #convert from samples to seocnds to minutes
        # fig,ax=plt.subplots()
        # ax.set_title('TDT and HDF clock alignment')
        # ax.set_xlabel('Time of Reward Period on TDT clock (min)')
        # ax.set_ylabel('HDF clock time (min)')
        # ax.plot(tdt_times,hdf_times,'o',alpha=0.5)
        # ax.plot([0,np.max(tdt_times)],[0,np.max(tdt_times)],'k--') #unity line
        # fig.suptitle(self.session)
        # xxx
        print('Alignment loaded!')

        return 
        
        
    def get_firing_rate(self):
        '''
        Count how many spikes occured in the time window of interest and then divide by window length.
        Time window defined as [time_align - t_before : time_align + t_after]

        Returns
        -------
        None.

        '''

        
        
        ## Get Spike Counts
        print('Binning and counting spikes..')
        firing_rates = np.zeros((self.num_trials,self.num_units))
        for trial in range(self.num_trials):
            
            win_begin = self.times_align[trial] - self.t_before
            win_end = self.times_align[trial] + self.t_after

            for i in range(self.num_units):
                
                unit_spikes = np.array(self.spike_times[i])
                num_spikes = sum( (unit_spikes>win_begin) & (unit_spikes<win_end) )
                
                firing_rates[trial,i] = num_spikes / (self.t_before + self.t_after)
                
        print('Done counting spikes!')

        self.firing_rate_df = pd.DataFrame(firing_rates,columns=self.unit_labels)
        # self.firing_rate_df['Trial'] = np.arange(self.num_trials)+1
        self.firing_rate_df['Unit_labels'] = [self.unit_labels for i in range(self.num_trials)] 
                
        return 
            
        


class ProcessLFP:
    '''
    From .mat, .hdf, and .ns2 files, load LFP times and calculate PSD for each trial
    
    Output: Pickle file with LFPsnippet and PSD for each unit for each trial
    '''
    
    def __init__(self, session: str):

        
        self.session = session
        self.file_prefix = os.path.join(DATA_FOLDER, self.session)
        
#         # [Initiate different data files]
#         self.ns2file = None
#         self.hdffile = None
#         self.matfile = None
#         self.pklfile = None
#         self.ns5file = None
#         self.nevfile = None
        
        ## Data
        self.has_hdf = False # For behavior
        self.has_ns5 = False # For syncing 
        self.has_ns2 = False # For LFP
        self.has_nev = False # For waveform 
        self.has_decoder = False # For decoder
        self.has_mat = False # For syncing
        self.has_pkl = False # For LFP snippets
        self.check_files()
        
        
        ## Get LFP snippets
        self.LFP_dict = None
        if self.has_pkl:
            self.load_LFP_snippets() 
        else:
            self.get_times_align()
            self.get_LFP_snippets() 
        

        
        

    def check_files(self):
        """
        For a proper analysis to be done,
        the hdf, nev, mat, and pkl files are essential.
        The ns5 and ns2 are optional.
        """
        
        if os.path.exists(self.file_prefix + '.hdf'):
            self.has_hdf = True
        if os.path.exists(self.file_prefix + '.ns5'):
            self.has_ns5 = True
        if os.path.exists(self.file_prefix + '.ns2'):
            self.has_ns2 = True
        if os.path.exists(self.file_prefix + '.nev'):
            self.has_nev = True
        if os.path.exists(self.file_prefix + '_syncHDF.mat'):
            self.has_mat = True
        if os.path.exists(self.file_prefix + '_nev_output.pkl'):
            self.has_nev_output = True
        if os.path.exists(self.file_prefix + '_KFDecoder.pkl'):
            self.has_decoder = True
        if os.path.exists(self.file_prefix + '_LFP.pkl'):
            self.has_pkl = True
        
          
        
    
        
    def get_times_align(self):
        '''
        Gets the array of indices (sample numbers) corresponding to the hold_center, 
        target, and check_reward time points of the given session.
        This facilitates time-aligned analyses.
        
        Parameters
        ----------
        hdf_files : list of hdf files for a single session
        syncHDF_files : list of syncHDF_files files which are used to make the alignment between behavior data and spike data
            
        Outputs
        -------
        target_hold_TDT_ind : 1D array containing the TDT indices for the target hold onset times of the given session
        '''
        print('Loading alignment..')
        
        assert self.has_hdf, FileNotFoundError(f'.hdf file not found! Session: {self.session}')
        assert self.has_mat, FileNotFoundError(f'.mat file not found! Session: {self.session}')
        
        self.hdf_file = self.file_prefix + '.hdf'
        self.syncHDF_file = self.file_prefix + '_syncHDF.mat'
        
        fs_hdf = 60 #hdf fs is always 60
        
        # load behavior data
        cb = Behavior.CenterOut([self.hdf_file]) #method needs hdf filenames in a list
        
        # align spike tdt times with hold center hdf indices using syncHDF files
        target_prompt_TDT_ind, DIO_freq = Behavior.get_HDFstate_TDT_LFPsamples(cb.ind_target_prompt,cb.state_time,self.syncHDF_file)
        target_hold_TDT_ind, DIO_freq = Behavior.get_HDFstate_TDT_LFPsamples(cb.ind_target_hold,cb.state_time,self.syncHDF_file)

        assert len(target_prompt_TDT_ind) == len(target_hold_TDT_ind)
        self.num_trials = len(target_hold_TDT_ind)

        self.start_mvmt_times = np.rint(target_prompt_TDT_ind / DIO_freq[0]).astype(int)
        self.end_mvmt_times = np.rint(target_hold_TDT_ind / DIO_freq[0]).astype(int)
        # self.fs = DIO_freq[0] #This is the ns5 fs
        
        # Behavior/task info
        self.deg = cb.target_loc_degs #for each trial
        self.block_type = cb.block_type
        
        
        
        # #Plot to check things out
        # tdt_times = target_hold_TDT_ind / DIO_freq / 60  #convert from samples to seocnds to minutes
        # hdf_times = (cb.state_time[ind_target_hold]) / fs_hdf / 60  #convert from samples to seocnds to minutes
        # fig,ax=plt.subplots()
        # ax.set_title('TDT and HDF clock alignment')
        # ax.set_xlabel('Time of Reward Period on TDT clock (min)')
        # ax.set_ylabel('HDF clock time (min)')
        # ax.plot(tdt_times,hdf_times,'o',alpha=0.5)
        # ax.plot([0,np.max(tdt_times)],[0,np.max(tdt_times)],'k--') #unity line
        # fig.suptitle(self.session)
        # xxx
        print('Alignment loaded!')

        return 
        
        
    def get_LFP_snippets(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        print('Getting LFP snippets for each trial...')
        
        assert self.has_ns2, FileNotFoundError(f'.ns2 file not found! Session: {self.session}')
        self.ns2 = NSFile(self.file_prefix + '.ns2')
        
        self.fs = self.ns2.get_file_data('ns2').parser.timestamp_resolution / self.ns2.get_file_data('ns2').parser.period
        
        assert self.fs == 1000.0
        
        snips = np.zeros( (self.num_trials,len(CHS),int(self.fs*SNIPPET_DURATION)) )
        trials = [] #trials used. trials which are longer than set duration.
        
        #convert from times to ns2 inds
        self.start_mvmt_inds = np.rint(self.start_mvmt_times * self.fs).astype(int)
        self.end_mvmt_inds = np.rint(self.end_mvmt_times * self.fs).astype(int)


        
        for trial in range(self.num_trials):
            # trtime_start = time.time()
            
            start_ind = self.start_mvmt_inds[trial] 
            end_ind = self.end_mvmt_inds[trial]
            
            if (end_ind - start_ind) < self.fs*SNIPPET_DURATION: #skip trials shorter than set duration
                continue
            
            trials.append(trial)
            
            
            
            for ch_idx,ch in enumerate(CHS):
                
                snip = self.ns2.get_file_data('ns2').parser.get_analog_data(
                    channel_index=ch,
                    start_index=start_ind,
                    index_count=int(self.fs*SNIPPET_DURATION)) #only take set duration amount of samples, regardless of end time
        
                snips[trial,ch_idx,:] = snip.copy()
        
        
            # trtime = np.rint(time.time() - trtime_start)
            # num_trs_todo = self.num_trials-trial-1
            # time_left = np.round(trtime*num_trs_todo/60,2) #mins
            # print(f'Trial {trial+1}/{self.num_trials} done. Approx time remaining: {time_left} mins.',end='\n')
             
        
        trials = np.array(trials)
        t = np.arange(0,SNIPPET_DURATION,step=1/self.fs)
        assert len(t) == np.shape(snips)[2]
        
        self.LFP_dict = {'session': self.session,
                        'LFP': snips,
                        't': t,
                        'fs': self.fs,
                        'chs': CHS,
                        'trial_nums': trials,
                        'trial_deg': self.deg[trials],
                        'trial_cond': self.block_type[trials],
                        #'pert_type': self.pert,
                        #'pert_mag': self.pert_mag
                        }
        
        with open(self.file_prefix+'_LFP.pkl','wb') as f:
            pickle.dump(self.LFP_dict,f)
            print(self.file_prefix+'_LFP.pkl saved!')
        
        return 
            
    def load_LFP_snippets(self):     
        
       # Load dict of previously saved LFP snippets
        with open(self.file_prefix+'_LFP.pkl','rb') as f:
            self.LFP_dict = pickle.load(f)
            print(self.session+'_LFP.pkl' + ' loaded!')

        return



class AnalyzeLFP():
    
    def __init__(self,LFP_dict):
        
        self.LFP_dict = LFP_dict
        self.session = LFP_dict['session']
        self.trial_nums = LFP_dict['trial_nums']
        self.has_PSD = False
        
        
    def _get_PSD(self):
        '''
        Gets the power spectral density (PSD) for each trial for each channel
        
        PSD [V**2] = (num_trials x num_chs x num_freqs)

        Returns
        -------
        None.

        '''
        
        print('Getting PSD..')
        
        first_run = True # to use to initialize PSD array on first run only
        
        for trial in range(len(self.trial_nums)):
            
            for ch in range(len(CHS)):
                
                f,psd = signal.welch(np.squeeze(self.LFP_dict['LFP'][trial,ch,:]),self.LFP_dict['fs'],
                              nperseg=256,noverlap=None,nfft=256*4, #to get spectral res of about 1Hz
                              scaling='spectrum') #units = V**2
                
                f_trim = (f>4) & (f<200)
                f = f[f_trim]
                psd = psd[f_trim]
                
                if first_run:
                    self.PSD = np.zeros((len(self.trial_nums),len(CHS),len(f)))
                    self.f = f
                    first_run = False
                    
                self.PSD[trial,ch,:] = psd
                self.has_PSD = True
                
                assert np.array_equal(self.f,f) #make sure freqs are equal throughout
                
        return
    

    def get_grandavg_band_powers(self,freq_band):
        
        if not self.has_PSD:
            self._get_PSD()
       
        #get desired frequencies
        band = (self.f > FREQ_BANDS[freq_band][0]) & (self.f < FREQ_BANDS[freq_band][1])
            
        self.band_power_dict = {'session':self.session,
                                'metric_name':freq_band + ' power',
                                'bl':np.mean(self.PSD[BL_TRIALS,:,band]),
                                'ep':np.mean(self.PSD[EP_TRIALS,:,band]),
                                'lp':np.mean(self.PSD[LP_TRIALS,:,band])}
 
        return self.band_power_dict
              

    def get_PSD_per_trialset(self):
        
        return


    def get_PSD_per_direction(self):
        
        self._get_PSD()
            
        all_degs = np.unique(self.LFP_dict['trial_deg'])

        avg = np.zeros((len(all_degs),len(self.f)))
        sem = np.zeros_like(avg)
        
        for i,deg in enumerate(all_degs):
            
            deg_trial_idxs = np.nonzero(self.LFP_dict['trial_deg'] == deg)

            avg[i,:] = np.mean(np.squeeze(self.PSD[deg_trial_idxs,:,:]),axis=(0,1)) #avg over chs and trials
            sem[i,:] = np.std(np.squeeze(self.PSD[deg_trial_idxs,:,:]),axis=(0,1)) / np.sqrt(len(deg_trial_idxs)) #sem with n=num_trials
            
        self.PSD_per_dir = {'session': self.session,
                            'deg': all_degs,
                            'avg': avg,
                            'sem': sem}        
        
        
        ## Plot
        
        fig,axs=plt.subplots(3,3)
        axs_list = [axs[0,0],axs[0,1],axs[0,2],
                    axs[1,0],         axs[1,2],
                    axs[2,0],axs[2,1],axs[2,2]]\
            
        axs[1,1].set_axis_off()
        
        for i,(ax,deg) in enumerate(zip(axs_list,all_degs)):
            
            ax.loglog(self.f,avg[i,:])
            # ax.fill_between(self.f,avg[i,:]+sem[i,:],avg[i,:]-sem[i,:])
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Freq (Hz)')
            ax.set_ylabel('PSD (V^2)')
            ax.set_title(f'{deg} degrees')
            
        fig.suptitle(self.session)
        fig.tight_layout()

        return                



 



    def get_WPLI(self,freq_band):
        '''
        Get the session averaged time-freq representation of connectivity for each area
        
        Load time-aligned LFP snippets for each session and compute connectivity 
        between brain areas according to the method specified by conn_method.
        This connectivity metric is computed for each channel combination before
        the grand average is computed. I.e. connectivity is computed for num_chs_area1 x num_chs_area2
        combinations before being averaged to yield a single grand average of connectivity
        btwn the two brain areas. These grand averages are then averaged across sessions.

        Parameters
        ----------
        - conn_method : str. E.g. 'coh', 'wpli', 'cacoh'
            Connectivity measure to compute.
            See https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html
            for full list of options
        - overwriteFlag: bool. 
            If true, will not load previously saved results. Will get results anew and save them, overwriting previously saved results.
            If false, will load previously save results (if available).        


        Does not return anything, but results in new object attributes:
        - t: 1D array of time points of time-freq representations of connectivity
        - f: 1D array of frequency bins of time-freq representations of connectivity
        - conn_eachsess: 3D array of trial-averaged time-freq representations of connectivity for each session.
            shape = num_sessions x len(f) x len(t)
        - conn_sessavg: 2D array of session-averaged time-freq representations of connectivity
            shape = len(f) x len(t)

        '''

        
        print('Getting WPLI..')
         
        #get all combinations of channels
        ch_combos = np.array(list(itertools.combinations(self.LFP_dict['chs'], 2)))
        sources = ch_combos[:,0]
        targets = ch_combos[:,1]
        
        fmin=4
        fmax=200
        
        self.wpli_dict = {'session':self.session,
                          'metric_name':freq_band + ' connectivity (WPLI)'}
        
        for trial_block,block in zip([BL_TRIALS,EP_TRIALS,LP_TRIALS], ['bl','ep','lp']):
         
            connectivity = spectral_connectivity_epochs(
                        data = self.LFP_dict['LFP'][trial_block,:,:], 
                        # names = session_data['chs']
                        method = 'wpli',
                        indices = (sources,targets),
                        sfreq = self.LFP_dict['fs'],
                        mode = 'fourier',
                        fmin = fmin, 
                        fmax = fmax,
                        # cwt_freqs = freqs,
                        # cwt_n_cycles = freqs / 4,
                        verbose = 'CRITICAL',
                        )
                     
            self.f = np.array(connectivity.freqs)
            self.wpli = np.nanmean(connectivity.get_data(),axis=0) #avg across all ch combos
            
            #get desired frequencies
            band = (self.f > FREQ_BANDS[freq_band][0]) & (self.f < FREQ_BANDS[freq_band][1])
                
            self.wpli_dict[block] = np.mean(self.wpli[band])
 
    
 
        # fig,ax = plt.subplots()
        # ax.loglog(self.f,self.wpli)
        
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.set_xlabel('Freq (Hz)')
        # ax.set_ylabel('Connectivity (WPLI)')
        
        # fig.suptitle(self.session)
        # fig.tight_layout()
        
        return self.wpli_dict
        


    
    
    
def plot_metric_longitudinally(list_of_metric_dicts):
    '''
    

    Parameters
    ----------
    list_of_metric_dicts : a list of dicts. Each dict must have the following fields:
                                session: str
                                metric_name: str
                                bl: float
                                ep: float
                                lp: float

    Returns
    -------
    None.

    '''    
    metric = list_of_metric_dicts[0]['metric_name']
    
    bl=[] #baseline results
    ep=[] #early perturbation results
    lp=[] #late perturbation results
    sessions=[] #xlabels
    
    #unpack results
    for metric_dict in list_of_metric_dicts:
        bl.append(metric_dict['bl'])
        ep.append(metric_dict['ep'])
        lp.append(metric_dict['lp'])
        sessions.append(metric_dict['session'][4:4+8]) #only take date
        assert metric_dict['metric_name'] == metric
        
    num_sessions = len(sessions)
    x=np.arange(num_sessions)
    
    #plot results
    fig,ax=plt.subplots()
    ax.plot(x,bl,label='BL',color='tab:blue')
    ax.plot(x,ep,label='EP',color='tab:orange')
    ax.plot(x,lp,label='LP',color='tab:green')
    
    ax.set_ylabel(metric)
    ax.set_xlabel('Time (sessions)')
    ax.set_xticks(x,sessions,rotation=-45,fontsize=SMALL_FONTSIZE)
    ax.set_title(f'Longitudinal tracking of {metric}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    
        

def plot_metric_grid(metric,block,theta_list,alpha_list,beta_list,low_gamma_list,high_gamma_list):        
    
    #normalize to % of first session
    theta_norm = np.array(theta_list)/theta_list[0] * 100
    alpha_norm = np.array(alpha_list)/alpha_list[0] * 100
    beta_norm = np.array(beta_list)/beta_list[0] * 100
    low_gamma_norm = np.array(low_gamma_list)/low_gamma_list[0] * 100
    high_gamma_norm = np.array(high_gamma_list)/high_gamma_list[0] * 100
    
    assert len(theta_norm) == len(SESSIONS)

    x = np.arange(len(SESSIONS)) #time axis
    y = np.arange(5) #freq axis (5 freq bins)

    X,Y = np.meshgrid(x,y)

    Z = np.array([theta_norm,alpha_norm,beta_norm,low_gamma_norm,high_gamma_norm])

    fig,ax=plt.subplots()
    c = ax.pcolormesh(X,Y,Z, vmin=-100,vmax=300,cmap='bwr')
    fig.colorbar(c,ax=ax,label='% of 1st session')
    
    sessions = [session[4:4+8] for session in SESSIONS]
    
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Time (sessions)')
    ax.set_xticks(x,sessions,rotation=-45,fontsize=SMALL_FONTSIZE)
    ax.set_yticks(y,['theta','alpha','beta','low gamma','high gamma'],fontsize=SMALL_FONTSIZE)
    ax.set_title(f'{metric} over freq and time, {block} trials only.')
    fig.tight_layout()

    return
    
    

#### Run

def run_PSD_per_dir():
    
    for session in SESSIONS:
        
        lfp = ProcessLFP(session)
        
        AnalyzeLFP(lfp.LFP_dict).get_PSD_per_direction()
        
    return

  
        
def run_WPLI(freq_band_name):
    
    metric_list = []
    
    for i,session in enumerate(SESSIONS):
        
        lfp = ProcessLFP(session)
        
        wpli_dict = AnalyzeLFP(lfp.LFP_dict).get_WPLI(freq_band_name)
        
        metric_list.append(wpli_dict)
        
        print(f'{session} done! ({i+1}/{len(SESSIONS)})\n')
        
    plot_metric_longitudinally(metric_list)
    
    print(f'{freq_band_name} WPLI done for {len(SESSIONS)} sessions!\n\n\n')
    
    return
        

def run_bandpower(freq_band_name):
    
    metric_list = []
    
    for i,session in enumerate(SESSIONS):
        
        lfp = ProcessLFP(session)
        
        band_power_dict = AnalyzeLFP(lfp.LFP_dict).get_grandavg_band_powers(freq_band_name)
        
        metric_list.append(band_power_dict)
        
        print(f'{session} done! ({i+1}/{len(SESSIONS)})\n')
        
    plot_metric_longitudinally(metric_list)
    
    print(f'{freq_band_name} power done for {len(SESSIONS)} sessions!\n\n\n')
    
    return
        

def bandpower_grid(block):
    
    theta_list = []
    alpha_list = []
    beta_list = []
    low_gamma_list = []
    high_gamma_list = []
    
    for i,session in enumerate(SESSIONS):
        
        lfp = ProcessLFP(session)
        
        lfp_analysis = AnalyzeLFP(lfp.LFP_dict)
        
        band_power_dict = lfp_analysis.get_grandavg_band_powers('theta')
        theta_list.append(band_power_dict[block])
        band_power_dict = lfp_analysis.get_grandavg_band_powers('beta')
        beta_list.append(band_power_dict[block])
        band_power_dict = lfp_analysis.get_grandavg_band_powers('alpha')
        alpha_list.append(band_power_dict[block])
        band_power_dict = lfp_analysis.get_grandavg_band_powers('low gamma')
        low_gamma_list.append(band_power_dict[block])
        band_power_dict = lfp_analysis.get_grandavg_band_powers('high gamma')
        high_gamma_list.append(band_power_dict[block])
        
        print(f'{session} done! ({i+1}/{len(SESSIONS)})\n')
        
    plot_metric_grid('PSD',block,theta_list,alpha_list,beta_list,low_gamma_list,high_gamma_list)
    
    print(f'Band power grid done for {len(SESSIONS)} sessions!\n\n\n')
    
    return
    

def WPLI_grid(block):
    
    theta_list = []
    alpha_list = []
    beta_list = []
    low_gamma_list = []
    high_gamma_list = []
    
    for i,session in enumerate(SESSIONS):
        
        lfp = ProcessLFP(session)
        
        lfp_analysis = AnalyzeLFP(lfp.LFP_dict)
        
        band_power_dict = lfp_analysis.get_WPLI('theta')
        theta_list.append(band_power_dict[block])
        band_power_dict = lfp_analysis.get_WPLI('beta')
        beta_list.append(band_power_dict[block])
        band_power_dict = lfp_analysis.get_WPLI('alpha')
        alpha_list.append(band_power_dict[block])
        band_power_dict = lfp_analysis.get_WPLI('low gamma')
        low_gamma_list.append(band_power_dict[block])
        band_power_dict = lfp_analysis.get_WPLI('high gamma')
        high_gamma_list.append(band_power_dict[block])
        
        print(f'{session} done! ({i+1}/{len(SESSIONS)})\n')
        
    plot_metric_grid('WPLI',block,theta_list,alpha_list,beta_list,low_gamma_list,high_gamma_list)
    
    print(f'Band power grid done for {len(SESSIONS)} sessions!\n\n\n')
    
    return



# run_bandpower('theta')
# run_bandpower('alpha')
# run_bandpower('beta')
# run_bandpower('gamma')
# run_bandpower('low gamma')
# run_bandpower('high gamma')
# run_WPLI('theta')
# run_WPLI('alpha')
# run_WPLI('beta')
# run_WPLI('gamma')
# run_WPLI('low gamma')
# run_WPLI('high gamma')
# bandpower_grid('bl')
# bandpower_grid('ep')
# bandpower_grid('lp')
WPLI_grid('bl')
WPLI_grid('ep')
WPLI_grid('lp')

        