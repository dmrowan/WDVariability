#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import argparse
from astropy.stats import LombScargle
import collections
from gPhoton import gphoton_utils
import heapq
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import _pickle as pickle
from progressbar import ProgressBar
import WDranker_2
import WDutils
import random

class Visit:
    def __init__(self, df):
        self.df = df
        self.timereset = False
        self.df = self.df.reset_index(drop=True)

    #Simple check if we have enough usable data
    def good_df(self):
        if len(self.df['t1']) == 0:
            return False
        else:
            exposuretup = WDranker_2.find_cEXP(self.df)
            self.exposure = exposuretup.exposure
            self.tflagged = exposuretup.t_flagged
            if (self.tflagged > (self.exposure / 4)) or (self.exposure < 500):
                return False
            else:
                return True
    #Reset time to minutes from start
    def reset_time(self):
        for i in range(len(self.df['t_mean'])):
            jd = gphoton_utils.calculate_jd(self.df['t_mean'][i])
            if i == 0:
                tmin = jd
            newtime = (jd - tmin) * 1440
            self.df.loc[i, 't_mean'] = newtime
        self.timereset = True

    #Inject an optical lc and scale by multiplicative factor
    def inject(self, observation_list, mf, plot=False):
        self.reset_time()
        t_mean = self.df['t_mean']
        #Shift to relative flux scales
        relativetup = WDutils.relativescales(self.df)
        flux = relativetup.flux
        flux_err = relativetup.err

        #Get optical data
        df_optical = selectOptical(observation_list, 
                                   exposure=self.exposure/60)

        optical_flux = np.interp(t_mean, df_optical['time'], 
                                 df_optical['flux'])

        optical_flux = optical_flux * mf

        self.flux_injected = [ flux[i] + optical_flux[i] 
                              for i in range(len(flux)) ]

        if plot:
            fig, ax = plt.subplots(1, 1)
            ax.errorbar(t_mean, self.flux_injected, 
                        yerr=flux_err, color='red')
            ax = WDutils.plotparams(ax)
            plt.show()
    
    #Use the periodogram metric to test if injected signal is recoverable
    def assessrecovery(self):
        time_seconds = self.df['t_mean'] * 60
        ls = LombScargle(time_seconds, self.flux_injected)
        freq, amp = ls.autopower(nyquist_factor=1)

        detrad = self.df['detrad']
        ls_detrad = LombScargle(time_seconds, detrad)
        freq_detrad, amp_detrad = ls_detrad.autopower(nyquist_factor=1)
        pgram_tup = WDranker_2.find_cPGRAM(ls, amp_detrad, 
                                           exposure=self.exposure)
        c_periodogram = pgram_tup.c
        ditherperiod_exists = pgram_tup.ditherperiod_exists
        if (not ditherperiod_exists) and (c_periodogram > 0):
            return True
        else:
            return False

    #See if we have an exisiting period
    def existingperiods(self):
        if not self.timereset:
            self.reset_time()

        time_seconds = self.df['t_mean'] * 60
        relativetup = WDutils.relativescales(self.df)
        flux = relativetup.flux
        ls = LombScargle(time_seconds, flux)
        freq, amp = ls.autopower(nyquist_factor=1)

        detrad = self.df['detrad']
        ls_detrad = LombScargle(time_seconds, detrad)
        freq_detrad, amp_detrad = ls_detrad.autopower(nyquist_factor=1)

        pgram_tup = WDranker_2.find_cPGRAM(ls, amp_detrad,
                                           exposure=self.exposure)
        c_periodogram = pgram_tup.c
        if c_periodogram > 0:
            return True
        else:
            return False


#Generate list of sources used for planet injection
def genMagLists(plot=False):
    print("---Generating Source Lists---")
    #Initialize lists
    mag_list = []
    path_list = []
    pbar = ProgressBar()
    #Iterate through NUV files in cwd
    for filename in pbar(os.listdir(os.getcwd())):
        if filename.endswith('.csv') and 'NUV' in filename:
            alldata = pd.read_csv(filename)
            alldata = WDutils.df_fullreduce(alldata)
            alldata = WDutils.tmean_correction(alldata)
            mag = np.nanmedian(alldata['mag_bgsub'])
            data = WDutils.dfsplit(alldata, 100)
            #See if we have any good visits
            for df in data:
                if len(df['t1']) == 0:
                    continue
                else:
                    exposuretup = WDranker_2.find_cEXP(df)
                    exposure = exposuretup.exposure
                    tflagged = exposuretup.t_flagged
                    if (tflagged > (exposure / 4)) or (exposure < 500):
                        continue
                    else:
                        mag_list.append(mag)
                        path_list.append(filename)
                        break
    median = np.median(mag_list)
    print(median)

    #Magnitude Histogram
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(12, 12))
        bins = np.arange(min(mag_list), max(mag_list)+.1, .1)
        ax.hist(mag_list, bins=bins, color='xkcd:red', linewidth=1.2, 
                edgecolor='black')
        ax = WDutils.plotparams(ax)

        fig.savefig("/home/dmrowan/WhiteDwarfs/NUVmaghist.pdf")

    #Save pickle
    mag_array = np.array(mag_list)
    path_array = np.array(path_list)
    tup = (mag_array,path_array)
    with open('MagList.pickle', 'wb') as handle:
        pickle.dump(tup, handle)


def selectLC(binvalue, mag_array, path_array):
    #First need to find indicies for bin
    binupper = binvalue + .1
    idx_bin = np.where( (mag_array >= binvalue)
                       &(mag_array < binupper) )[0]
    filename = np.random.choice(path_array[idx_bin])
    
    #Standard data reduction procedure
    assert(os.path.isfile(filename))
    usecols = ['t0', 't1', 't_mean',
               'cps_bgsub', 'cps_bgsub_err', 'counts',
               'flux_bgsub', 'flux_bgsub_err',
               'detrad', 'flags', 'exptime']
    alldata = pd.read_csv(filename, usecols=usecols)
    alldata = WDutils.df_fullreduce(alldata)
    alldata = WDutils.tmean_correction(alldata)
    #Split into visits 
    data = WDutils.dfsplit(alldata, 100)
    #Pull good visits
    visit_list = []
    for df in data:
        visit = Visit(df)
        if visit.good_df() == True: 
            if visit.existingperiods() == False:
                visit_list.append(visit)
    #Select random visit
    visit = random.choice(visit_list)
    return visit


#Select random observation and time chunk
def selectOptical(observation_list, plot=False, exposure=30):
    exphalf = exposure/2
    opticalLC = random.choice(observation_list)
    maxtime = max(opticalLC['time'])
    idx_low = np.where(opticalLC['time'] < exphalf)[0][-1]
    idx_high = np.where(opticalLC['time'] > maxtime-exphalf)[0][0]
    idx_center = np.arange(idx_low, idx_high, 1)
    time_center = np.random.choice(opticalLC['time'][idx_center])

    time_center = 121 #Setting this value for testing

    df_optical = opticalLC[(opticalLC['time'] > time_center-exphalf) 
                          &(opticalLC['time'] < time_center+exphalf)]
    df_optical = df_optical.reset_index(drop=True)
    tmin = df_optical['time'][0]
    for i in range(len(df_optical['time'])):
        df_optical.loc[i, 'time'] = df_optical['time'][i] - tmin
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(20, 6))
        ax.scatter(df_optical['time'], df_optical['flux'])
        ax = WDutils.plotparams(ax)
        plt.show()
    return df_optical

def main(mag, mag_array, path_array, observation_list):
    visit = selectLC(mag, mag_array, path_array)
    mf = random.choice(np.arange(0, 2, .01))
    visit.inject(observation_list, mf)
    result = visit.assessrecovery()
    tup = (mag, mf, result)
    return tup

def wrapper(mag_array, path_array, observation_list, iterations=1):
    bins = np.arange(min(mag_array), max(mag_array)+.1, .1)
    result_list = []
    pool = mp.Pool(processes = mp.cpu_count()+2)
    jobs=[]
    result_list=[]
    for i in iterations:
        for mag in bins:
            job = pool.apply_async(main, args=(mag, mag_array,
                                               path_array, 
                                               observation_list,))
            jobs.append(job)
    for job in jobs:
        result_list.append(job.get())
    #Do something to save the results


if __name__ == '__main__':
    
    with open('fakeLC.pickle', 'rb') as p:
        LC = pickle.load(p)
    observation_list = [LC]
    #selectOptical(observation_list, plot=True, exposure=22)
    visit = selectLC(19, np.array([19]), 
                     np.array(
                         ['GalexData_run6/SDSS-J222816.29+134714.4-NUV.csv']))
    visit.inject(observation_list, 1, plot=True)
    result = visit.assessrecovery()
    print(result)














