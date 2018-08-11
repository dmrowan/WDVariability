#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import argparse
from astropy.stats import LombScargle
from gPhoton import gphoton_utils
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import _pickle as pickle
from progressbar import ProgressBar
import random
import time
import WDranker_2
import WDutils

desc="""
WD_Recovery: Procedure for injecting and recovering synthetic optical 
data from WD-1145.
"""

#Define a class for a galex visit df
class Visit:
    def __init__(self, df):
        self.df = df
        self.timereset = False
        self.df = self.df.reset_index(drop=True)

    #Simple check if we have enough usable data
    def good_df(self):
        #Create a second data frame with fully reduced data
        #Use this as a way to check flag points
        df2 = WDutils.df_fullreduce(self.df)
        if len(df2['t1']) <= 10:
            return False

        elif len(self.df['t1']) == 0:
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
    def inject(self, opticalLC, mf, plot=False):
        if self.timereset==False:
            self.reset_time()

        t_mean = self.df['t_mean']
        #Shift to relative flux scales
        relativetup = WDutils.relativescales_1(self.df)
        flux = relativetup.flux
        flux_err = relativetup.err
        
        #Get optical data
        df_optical = selectOptical(opticalLC, 
                                   exposure=self.exposure/60, plot=plot)

        #linear interpolation to match times
        optical_flux = np.interp(t_mean, df_optical['time'], 
                                 df_optical['flux'])

        #Scale by multiplicative factor
        optical_flux = optical_flux * mf
        flux_injected = [ flux[i] * optical_flux[i] 
                              for i in range(len(flux)) ]

        #Put back into df
        self.df.loc[:, 'flux_bgsub'] = flux_injected
        self.df.loc[:, 'flux_bgsub_err'] = flux_err
        #Remove colored points (flag, expt, sigma clip)
        coloredtup = WDutils.ColoredPoints(self.df)
        droppoints = np.unique(np.concatenate([coloredtup.redpoints, 
                                               coloredtup.bluepoints]))
        self.df = self.df.drop(index=droppoints)
        self.df = self.df.reset_index(drop=True)

        t_mean = self.df['t_mean']
        self.flux_injected = self.df['flux_bgsub']
        flux_err = self.df['flux_bgsub_err']

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
        #Return 0,1 rseult of recovery
        c_periodogram = pgram_tup.c
        ditherperiod_exists = pgram_tup.ditherperiod_exists
        if (not ditherperiod_exists) and (c_periodogram > 0):
            return 1
        else:
            return 0

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
            #Do a full data reduction
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

    #Save as pickle
    mag_array = np.array(mag_list)
    path_array = np.array(path_list)
    tup = (mag_array,path_array)
    with open('MagList.pickle', 'wb') as handle:
        pickle.dump(tup, handle)


#Choose a source and visit
def selectLC(binvalue, mag_array, path_array):
    #First need to find indicies for bin
    binupper = binvalue + .1
    idx_bin = np.where( (mag_array >= binvalue)
                       &(mag_array < binupper) )[0]
    filename = np.random.choice(path_array[idx_bin])
    print(filename)
    
    #Standard data reduction procedure
    assert(os.path.isfile(filename))
    usecols = ['t0', 't1', 't_mean',
               'mag_bgsub',
               'cps_bgsub', 'cps_bgsub_err', 'counts',
               'flux_bgsub', 'flux_bgsub_err',
               'detrad', 'flags', 'exptime']
    alldata = pd.read_csv(filename, usecols=usecols)
    #Data reduction and time correction
    alldata = WDutils.df_reduce(alldata)
    alldata = WDutils.tmean_correction(alldata)
    #Split into visits 
    data = WDutils.dfsplit(alldata, 100)
    source_mag = round(np.nanmedian(alldata['mag_bgsub']),5)
    #Pull good visits
    visit_list = []
    for df in data:
        visit = Visit(df)
        if visit.good_df() == True: 
            if visit.existingperiods() == False:
                print
                visit_list.append(visit)

    #If there were no good visits, pick a new source
    if len(visit_list) == 0:
        print(filename, "no sources in visit list")
        visit, source_mag = selectLC(binvalue, mag_array, path_array)
        return visit, source_mag
    #Select random visit
    else:
        visit = random.choice(visit_list)
        return visit, source_mag


#Select random observation and time chunk
def selectOptical(opticalLC, plot=False, exposure=30):
    exphalf = exposure/2
    maxtime = max(opticalLC['time'])
    #Find idx range corresponding to exposure
    idx_low = np.where(opticalLC['time'] < exphalf)[0][-1]
    idx_high = np.where(opticalLC['time'] > maxtime-exphalf)[0][0]
    idx_center = np.arange(idx_low, idx_high, 1)
    time_center = np.random.choice(opticalLC['time'][idx_center])

    df_optical = opticalLC[(opticalLC['time'] > time_center-exphalf) 
                          &(opticalLC['time'] < time_center+exphalf)]
    df_optical = df_optical.reset_index(drop=True)
    #Reset time to start from zero
    tmin = df_optical['time'][0]
    for i in range(len(df_optical['time'])):
        df_optical.loc[i, 'time'] = df_optical['time'][i] - tmin
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(20, 6))
        ax.scatter(df_optical['time'], df_optical['flux'])
        ax = WDutils.plotparams(ax)
        plt.show()
    return df_optical

#Full injection procedure for single source
def main(mag, mag_array, path_array, opticalLC, fname=None):
    #Select a visit
    visit, source_mag = selectLC(mag, mag_array, path_array)

    #Select a multiplicative factor
    mf = random.choice(np.arange(0, 2, .1))
    mf = round(mf, 1) #np.arange has some odd behaviors
    visit.inject(opticalLC, mf)
    result = visit.assessrecovery()
    tup = (mag, source_mag, mf, result)

    #Generate output string
    outputstr = (str(source_mag)+","
                +str(mf)+","
                +str(result))

    #Append to outputfile
    if fname is not None:
        os.system("echo {0} >> {1}".format(outputstr, fname))

    print(outputstr)
    return tup

def wrapper(mag_array, path_array, opticalLC, 
            iterations, ml, mu, bs, p, fname):
    #Set up magnitude bins
    minbin=ml
    maxbin=mu
    bins = np.arange(minbin, maxbin+bs, bs)
    bins = np.array([ round(b, 1) for b in bins ])

    #Create mp pool and iterate
    pool = mp.Pool(processes = p)
    jobs=[]
    print(iterations)
    for i in range(iterations):
        for mag in bins:
            job = pool.apply_async(main, args=(mag, mag_array, 
                                               path_array, opticalLC, 
                                               fname,))
            jobs.append(job)
    for job in jobs:
        job.get()

#Iterate through all sources and all visits 
def testfunction(path_array):
    usecols = ['t0', 't1', 't_mean',
               'mag_bgsub',
               'cps_bgsub', 'cps_bgsub_err', 'counts',
               'flux_bgsub', 'flux_bgsub_err',
               'detrad', 'flags', 'exptime']
    for filename in path_array:
        assert(os.path.isfile(filename))
        alldata = pd.read_csv(filename, usecols=usecols)
        #Data reduction and time correction
        alldata = WDutils.df_reduce(alldata)
        alldata = WDutils.tmean_correction(alldata)
        #Split into visits 
        data = WDutils.dfsplit(alldata, 100)
        source_mag = round(np.nanmedian(alldata['mag_bgsub']),5)
        visit_list = []
        for df in data:
            visit = Visit(df)
            if visit.good_df() == True: 
                if visit.existingperiods() == False:
                    print
                    visit_list.append(visit)

        #If there were no good visits, pick a new source
        if len(visit_list) == 0:
            print(filename, "no sources in visit list")
            continue
        for visit in visit_list:
            mf = random.choice(np.arange(0, 2, .1))
            mf = round(mf, 1) #np.arange has some odd behaviors
            try:
                visit.inject(opticalLC, mf)
                result = visit.assessrecovery()
            except:
                result = "Error with source "+filename

            #Generate output string
            outputstr = (str(source_mag)+","
                        +str(mf)+","
                        +str(result))

            print(outputstr)
            fname = 'testoutput.txt'
            os.system("echo {0} >> {1}".format(outputstr, fname))

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--iter", help="Number of iterations to run", 
                        type=int, default=100)
    parser.add_argument("--ml", help="Lower mag range", 
                        type=float, default=16)
    parser.add_argument("--mu", help="Upper mag range", 
                        type=float, default=21)
    parser.add_argument("--bs", help="Magnitude bin size", 
                        type=float, default=.1)
    parser.add_argument("--p", help="Number of processes to spawn", 
                        type=int, default=4)
    parser.add_argument("--output", help="Output filename to save results",
                        type=str, default=None)
    args=parser.parse_args()

    #Argument assertions
    assert(args.iter > 0)
    assert( (args.ml > 10) and (args.ml < 30) and (args.ml < args.mu) )
    assert( (args.mu > 10) and (args.mu < 30) )
    assert( (args.bs >= .1) and (args.bs <=2 ) )
    assert(args.p >= 1)

    #Pickle loads
    with open('1145LC.pickle', 'rb') as p:
        opticalLC = pickle.load(p)
    with open('MagList.pickle', 'rb') as p2:
        MagList = pickle.load(p2)
    
    mag_array = MagList[0]
    path_array = MagList[1]

    #wrapper(mag_array, path_array, opticalLC, args.iter, args.ml,
    #        args.mu, args.bs, args.p, args.output)

    testfunction(path_array)
    


    #for i in range(50):
    #    visit, source_mag = selectLC(1, np.array([1]), np.array(['WD-0102+004-NUV.csv']))
    #    visit.inject(opticalLC, 1, plot=False)
    #    result = visit.assessrecovery()
    #    print(result)

