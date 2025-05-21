# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:35:03 2025

@author: coleb
"""
#### Define global things

## Paths
# DATA_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\Population Stability\Data"
DATA_FOLDER = r"D:\Population Stability\Data"
BMI_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\bmi_python"
# DATA_FOLDER = r"F:\cole"
# BMI_FOLDER = r"C:\Users\crb4972\Desktop\bmi_python"


import os
import sys
import time
import pickle
import itertools
import numpy as np
import pandas as pd
# from fooof import FOOOF
from scipy import signal
# import statsmodels.api as sm
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
# from matplotlib_venn import venn2,venn3
from mne_connectivity import spectral_connectivity_epochs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import Behavior
import Sessions
SESSIONS, GOOD_CH_DICT = Sessions.get_sessions()





TEST = False
# CHS = np.arange(0,128,step=8,dtype=int) #define subset of channels to use
CHS = GOOD_CH_DICT[SESSIONS[0]]
downsample_factor = 2
CHS = CHS[::2]
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


SMALL_FONTSIZE = 8
MED_FONTSIZE = 12
LARGE_FONTSIZE = 18




## Import Neuroshare libraries from bmi_python folder
NS_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns')
sys.path.insert(1,BMI_FOLDER) #add bmi_python folder to package search path
sys.path.insert(2, NS_FOLDER) #add neuroshare python folder to package search path
from nsfile import NSFile
os.chdir(DATA_FOLDER)

def cosine_model(theta, MD, PD, avg_pwr):
    """
    Cosine function for lfp power fitting.
    theta: angles (in radians)
    MD: amplitude of cosine
    PD: phase shift (preferred direction)
    meanFR: baseline power level
    """
    return MD * np.cos(theta - PD) + avg_pwr

def zscore(list_):
    return ( np.array(list_) - np.mean(list_) )/ np.std(list_)


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
        
        self.get_times_align()
        
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
                        # 'trial_cond': self.block_type[trials],
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

        if not (list(self.LFP_dict['chs']) == list(CHS)):
            print('Channel selection out of date! Must re-get snippets.')
            self.get_LFP_snippets()
        
        return



class AnalyzeLFP():
    
    def __init__(self,LFP_dict):
        
        self.LFP_dict = LFP_dict
        self.session = LFP_dict['session']
        self.num_trials = len(LFP_dict['trial_nums'])
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
        
        for trial in range(self.num_trials):
            
            for ch in range(len(CHS)):
                
                f,psd = signal.welch(np.squeeze(self.LFP_dict['LFP'][trial,ch,:]),self.LFP_dict['fs'],
                              nperseg=256,noverlap=None,nfft=256*4, #to get spectral res of about 1Hz
                              scaling='spectrum') #units = V**2
                
                f_trim = (f>2) & (f<200)
                f = f[f_trim]
                psd = psd[f_trim]
                
                if first_run:
                    self.PSD = np.full((self.num_trials,len(CHS),len(f)),np.nan)
                    self.f = f
                    first_run = False
                
                if np.max(psd) != 0:
                    self.PSD[trial,ch,:] = psd / np.max(psd) #normalize by max amplitude to make more comparable across different noise levels
                # else:
                #     print(f'Skipped cuz of invalid divide: trial {trial} ch {ch}')
                    
                assert np.array_equal(self.f,f) #make sure freqs are equal throughout
                
        self.has_PSD = True        
        return
    
    def get_band_power_per_trial(self,freq_band):
        ### DOES BASELINE TRIALS ONLY
        
        if not self.has_PSD:
            self._get_PSD()
       
        #get desired frequencies
        band = (self.f > FREQ_BANDS[freq_band][0]) & (self.f < FREQ_BANDS[freq_band][1])
 
        
        return zscore(np.nanmean(self.PSD[BL_TRIALS,:,band],axis=(1,2)))
    
    def get_LDA_params(self):
        ### DOES BASELINE TRIALS ONLY
        
        X = np.full((NUM_TRIALS_PER_BLOCK,len(FREQ_BANDS)),np.nan)
        
        for i,freq_band in enumerate(FREQ_BANDS):
            X[:,i] = self.get_band_power_per_trial(freq_band)
        
        y = self.LFP_dict['trial_deg'][:NUM_TRIALS_PER_BLOCK]
        print(np.unique(self.LFP_dict['trial_deg']))
        lda = LDA(solver='lsqr')
        lda.fit(X,y)
        print(lda.coef_)
        print(lda.classes_)
        xxx
        return
    
    def get_grandavg_band_powers(self,freq_band):
        
        if not self.has_PSD:
            self._get_PSD()
       
        #get desired frequencies
        band = (self.f > FREQ_BANDS[freq_band][0]) & (self.f < FREQ_BANDS[freq_band][1])
            
        self.band_power_dict = {'session':self.session,
                                'metric_name':freq_band + ' power',
                                'bl':np.nanmean(self.PSD[BL_TRIALS,:,band]),
                                'ep':np.nanmean(self.PSD[EP_TRIALS,:,band]),
                                'lp':np.nanmean(self.PSD[LP_TRIALS,:,band])}
 
        return self.band_power_dict


    def get_PSD_per_direction(self):
        
        self._get_PSD()
            
        all_degs = np.unique(self.LFP_dict['trial_deg'])

        avg = np.zeros((len(all_degs),len(self.f)))
        sem = np.zeros_like(avg)
        
        for i,deg in enumerate(all_degs):
            
            deg_trial_idxs = np.nonzero(self.LFP_dict['trial_deg'] == deg)

            avg[i,:] = np.nanmean(np.squeeze(self.PSD[deg_trial_idxs,:,:]),axis=(0,1)) #avg over chs and trials
            sem[i,:] = np.nanstd(np.squeeze(self.PSD[deg_trial_idxs,:,:]),axis=(0,1)) / np.sqrt(len(deg_trial_idxs)) #sem with n=num_trials
            
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


    def get_PSD_PD(self,freq_band_name):
        
        self._get_PSD()
        
        print('Finding tuning curve for each ch..')
        
        self.PDs = np.full((len(CHS)),np.nan)
        self.MDs = np.full_like(self.PDs,np.nan)
        self.rss = 0
        
        all_degs = np.unique(self.LFP_dict['trial_deg'])

        for ch in range(len(CHS)):
                
            #get desired frequencies
            band = (self.f > FREQ_BANDS[freq_band_name][0]) & (self.f < FREQ_BANDS[freq_band_name][1]) 
            
            pwr_per_dir = np.zeros((len(all_degs)))
            sem_per_dir = np.zeros_like(pwr_per_dir)
            
            #get power per direction
            for i,deg in enumerate(all_degs):
                
                deg_trial_idxs = self.LFP_dict['trial_deg'] == deg

                pwr_per_dir[i] = np.nanmean(self.PSD[np.ix_(deg_trial_idxs,[ch],band)]) #avg over trials and freq band
                sem_per_dir[i] = np.nanstd(self.PSD[np.ix_(deg_trial_idxs,[ch],band)]) / np.sqrt(sum(deg_trial_idxs)) #sem w/ n=num_trials
                
            
            ##fit PD tuning
            
            #initial guesses
            avg_pwr_guess = np.mean(pwr_per_dir)
            MD_guess = (np.max(pwr_per_dir) - np.min(pwr_per_dir)) / 2
            PD_guess = np.deg2rad(all_degs[np.argmax(pwr_per_dir)])
            
            initial_guesses = [ #add some jitter to the initial guess
                [MD_guess, PD_guess, avg_pwr_guess],
                [MD_guess * 1.1, PD_guess + (np.pi / 8), avg_pwr_guess * 1.1],
                [MD_guess * 0.9, PD_guess - (np.pi / 8), avg_pwr_guess * 0.9],
            ]
            
            # Initialize default values in case fitting fails
            MD, PD, meanFR, pred_fr = np.nan, np.nan, np.nan, np.nan
            
            #curve fit
            for initial_guess in initial_guesses:
                try:
                    (MD, PD, avg_pwr), pcov = curve_fit(
                        cosine_model, np.deg2rad(all_degs), pwr_per_dir,
                        p0=initial_guess,
                        bounds=([0, -2 * np.pi, -np.inf], [np.inf, 2 * np.pi, np.inf]))
                except RuntimeError:
                    continue  # Try the next initial guess if fitting fails
           
            if PD < 0:
                PD += 2*np.pi
                
            #calc error
            pred_pwr = cosine_model(np.deg2rad(all_degs), *(MD, PD, avg_pwr))
            self.rss += np.sum((pwr_per_dir - pred_pwr)**2)
            
            # #plot
            # fig,ax=plt.subplots()
            # ax.errorbar(all_degs,pwr_per_dir,sem_per_dir,fmt='o',label='actual')
            # theta_plot = np.linspace(0,2*np.pi,500)
            # ax.plot(np.rad2deg(theta_plot),cosine_model(theta_plot,MD,PD,avg_pwr),label=f'fit: PD = {np.rad2deg(PD).round()}, MD = {MD.round(2)}')
            # ax.legend()
            # ax.set_title(f'ch {ch}, {freq_band_name} power')
            # fig.suptitle(self.session)
                

                
            self.PDs[ch] = PD
            self.MDs[ch] = MD

        print('All preferred directions found!')
        
        return self.PDs            
 



    def get_WPLI_freq_band(self,freq_band):
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
        
    
    
    def get_WPLI_all_bands(self,block):
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
        

        if block=='bl':
            trials = BL_TRIALS
        elif block=='ep':
            trials = EP_TRIALS
        elif block=='lp':
            trials = LP_TRIALS
         
        connectivity = spectral_connectivity_epochs(
                    data = self.LFP_dict['LFP'][trials,:,:], 
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
        band = (self.f > FREQ_BANDS['theta'][0]) & (self.f < FREQ_BANDS['theta'][1])
        theta = np.mean(self.wpli[band])
        band = (self.f > FREQ_BANDS['alpha'][0]) & (self.f < FREQ_BANDS['alpha'][1])
        alpha = np.mean(self.wpli[band])
        band = (self.f > FREQ_BANDS['beta'][0]) & (self.f < FREQ_BANDS['beta'][1])
        beta = np.mean(self.wpli[band])
        band = (self.f > FREQ_BANDS['low gamma'][0]) & (self.f < FREQ_BANDS['low gamma'][1])
        low_gamma = np.mean(self.wpli[band])
        band = (self.f > FREQ_BANDS['high gamma'][0]) & (self.f < FREQ_BANDS['high gamma'][1])
        high_gamma = np.mean(self.wpli[band])
                
                
 
    
 
        # fig,ax = plt.subplots()
        # ax.loglog(self.f,self.wpli)
        
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.set_xlabel('Freq (Hz)')
        # ax.set_ylabel('Connectivity (WPLI)')
        
        # fig.suptitle(self.session)
        # fig.tight_layout()
        
        return theta,alpha,beta,low_gamma,high_gamma

    
    
    
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
    
    ax.set_ylabel(metric,fontsize=MED_FONTSIZE)
    ax.set_xlabel('Time (sessions)',fontsize=MED_FONTSIZE)
    ax.set_xticks(x,sessions,rotation=-45,fontsize=SMALL_FONTSIZE)
    ax.set_title(f'Longitudinal tracking of {metric}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    
        

def plot_metric_grid(zscore_flag,metric,block,theta_list,alpha_list,beta_list,low_gamma_list,high_gamma_list):        
    
    if zscore_flag:
        theta = zscore(theta_list)
        alpha = zscore(alpha_list)
        beta = zscore(beta_list)
        low_gamma = zscore(low_gamma_list)
        high_gamma = zscore(high_gamma_list)
        clabel='z-score PSD'
        vmin,vmax=-2.5,2.5
        
    else:
        theta = theta_list
        alpha = alpha_list
        beta = beta_list
        low_gamma = low_gamma_list
        high_gamma = high_gamma_list
        clabel='wpli'
        vmin,vmax=0,0.35
    
    assert len(theta) == len(SESSIONS)

    x = np.arange(len(SESSIONS)) #time axis
    y = np.arange(5) #freq axis (5 freq bins)

    X,Y = np.meshgrid(x,y)

    Z = np.array([theta,alpha,beta,low_gamma,high_gamma])

    fig,ax=plt.subplots()
    c = ax.pcolormesh(X,Y,Z, vmin=vmin,vmax=vmax,cmap='bwr')
    fig.colorbar(c,ax=ax,label=clabel)
    
    sessions = [session[8:12] for session in SESSIONS]
    
    ax.set_ylabel('Frequency',fontsize=MED_FONTSIZE)
    ax.set_xlabel('Time (sessions)',fontsize=MED_FONTSIZE)
    ax.set_xticks(x,sessions,rotation=-90,fontsize=SMALL_FONTSIZE)
    ax.set_yticks(y,['theta','alpha','beta','low gamma','high gamma'],fontsize=SMALL_FONTSIZE)
    ax.set_title(f'{metric} over freq and time, {block} trials only.')
    fig.tight_layout()

    return
    
    














#### Run methods


def run_getsnippets():
    for i,session in enumerate(SESSIONS):
        looptime_start = time.time()
        
        _ = ProcessLFP(session)
        
        # timer
        looptime = np.rint(time.time() - looptime_start)
        num_loops_todo = len(SESSIONS)-i-1
        time_left = np.round(looptime*num_loops_todo/60,2) #mins
        print(f'Session {i+1}/{len(SESSIONS)} done. Approx time remaining: {time_left} mins.',end='\n')
    return
    
    
def run_PSD_PD(freq_band_name):
    
    
    all_PDs = np.zeros((len(CHS),len(SESSIONS)))
    
    for i,session in enumerate(SESSIONS):
        looptime_start = time.time()
        
        lfp = ProcessLFP(session)
        
        sess_PD = AnalyzeLFP(lfp.LFP_dict).get_PSD_PD(freq_band_name)
        
        all_PDs[:,i] = np.rad2deg(sess_PD)
        
        # timer
        looptime = np.rint(time.time() - looptime_start)
        num_loops_todo = len(SESSIONS)-i-1
        time_left = np.round(looptime*num_loops_todo/60,2) #mins
        print(f'Session {i+1}/{len(SESSIONS)} done. Approx time remaining: {time_left} mins.',end='\n')
        
    PD_diff = np.diff(all_PDs,axis=1) #difference between sessions for each channel
    
    #find miminum angle (e.g. 350 -> -10)
    PD_diff[PD_diff>180] -= 360 
    PD_diff[PD_diff<-180] += 360 
    
    fig,ax = plt.subplots()
    ax.hist(PD_diff[~np.isnan(PD_diff)].flatten())
    ax.set_title(f'Difference in PD between adjacent sessions: {freq_band_name} power')
    ax.set_xlabel('PD Difference (deg)')
    ax.set_ylabel('Count')
    
        
    return

  
        
def run_WPLI(freq_band_name):
    
    metric_list = []
    
    for i,session in enumerate(SESSIONS):
        
        looptime_start = time.time()
        
        lfp = ProcessLFP(session)
        wpli_dict = AnalyzeLFP(lfp.LFP_dict).get_WPLI(freq_band_name)
        metric_list.append(wpli_dict)
        print(f'{session} done!')
        
        # timer
        looptime = np.rint(time.time() - looptime_start)
        num_loops_todo = len(SESSIONS)-i-1
        time_left = np.round(looptime*num_loops_todo/60,2) #mins
        print(f'Session {i+1}/{len(SESSIONS)} done. Approx time remaining: {time_left} mins.',end='\n')
        
    plot_metric_longitudinally(metric_list)
    
    print(f'{freq_band_name} WPLI done for {len(SESSIONS)} sessions!\n\n\n')
    
    return
        

def run_bandpower(freq_band_name):
    
    metric_list = []
    
    for i,session in enumerate(SESSIONS):
        
        looptime_start = time.time()
        
        # try:
        if True: #to maintain indent when try commented out
            lfp = ProcessLFP(session)
            band_power_dict = AnalyzeLFP(lfp.LFP_dict).get_grandavg_band_powers(freq_band_name)
            metric_list.append(band_power_dict)
            print(f'{session} done!')
          
        # skip sessions that throw an error, but print out error
        # except BaseException as err:
        #     print("*"*20)
        #     print(f"Unexpected {err=}, {type(err)=}")
        #     print(f'{session} skipped!')
        #     print("*"*20)
        
        # timer
        looptime = np.rint(time.time() - looptime_start)
        num_loops_todo = len(SESSIONS)-i-1
        time_left = np.round(looptime*num_loops_todo/60,2) #mins
        print(f'Session {i+1}/{len(SESSIONS)} done. Approx time remaining: {time_left} mins.',end='\n')
        
    plot_metric_longitudinally(metric_list)
    
    print(f'{freq_band_name} power done for {len(SESSIONS)} sessions!\n\n\n')
    
    return
        

def run_bandpower_grid(block):
    
    theta_list = []
    alpha_list = []
    beta_list = []
    low_gamma_list = []
    high_gamma_list = []
    
    for i,session in enumerate(SESSIONS):
        
        looptime_start = time.time()
            
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
    
        print(f'{session} done!')
        
        # timer
        looptime = np.rint(time.time() - looptime_start)
        num_loops_todo = len(SESSIONS)-i-1
        time_left = np.round(looptime*num_loops_todo/60,2) #mins
        print(f'Session {i+1}/{len(SESSIONS)} done. Approx time remaining: {time_left} mins.',end='\n')
        
    plot_metric_grid(True,'PSD',block,theta_list,alpha_list,beta_list,low_gamma_list,high_gamma_list)
    
    print(f'Band power grid done for {len(SESSIONS)} sessions!\n\n\n')
    
    return
    

def run_WPLI_grid(block):
    
    theta_list = []
    alpha_list = []
    beta_list = []
    low_gamma_list = []
    high_gamma_list = []
    
    for i,session in enumerate(SESSIONS):
        
        looptime_start = time.time()
            
        lfp = ProcessLFP(session)
        lfp_analysis = AnalyzeLFP(lfp.LFP_dict)
        
        theta,alpha,beta,low_gamma,high_gamma = lfp_analysis.get_WPLI_all_bands(block)
        theta_list.append(theta)
        alpha_list.append(alpha)
        beta_list.append(beta)
        low_gamma_list.append(low_gamma)
        high_gamma_list.append(high_gamma)
    
        print(f'{session} done!')
        
        # timer
        looptime = np.rint(time.time() - looptime_start)
        num_loops_todo = len(SESSIONS)-i-1
        time_left = np.round(looptime*num_loops_todo/60,2) #mins
        print(f'Session {i+1}/{len(SESSIONS)} done. Approx time remaining: {time_left} mins.',end='\n')
        
    plot_metric_grid(False,'WPLI',block,theta_list,alpha_list,beta_list,low_gamma_list,high_gamma_list)
    
    print(f'Band power grid done for {len(SESSIONS)} sessions!\n\n\n')
    
    return


def run_LDA():
    
    for i,session in enumerate(SESSIONS):
        
        looptime_start = time.time()
            
        lfp = ProcessLFP(session)
        params = AnalyzeLFP(lfp.LFP_dict).get_LDA_params()
   
        print(f'{session} done!')
        
        # timer
        looptime = np.rint(time.time() - looptime_start)
        num_loops_todo = len(SESSIONS)-i-1
        time_left = np.round(looptime*num_loops_todo/60,2) #mins
        print(f'Session {i+1}/{len(SESSIONS)} done. Approx time remaining: {time_left} mins.',end='\n')




#### Run

# run_getsnippets()

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

# run_bandpower_grid('bl')
# run_bandpower_grid('ep')
# run_bandpower_grid('lp')
# run_WPLI_grid('bl')
# run_WPLI_grid('ep')
# run_WPLI_grid('lp')

# run_PSD_PD('theta')
# run_PSD_PD('alpha')
run_PSD_PD('beta')
# run_PSD_PD('gamma')
# run_PSD_PD('low gamma')
# run_PSD_PD('high gamma')

# run_LDA()

        