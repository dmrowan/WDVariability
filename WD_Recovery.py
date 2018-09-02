#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import argparse
from astropy.stats import LombScargle
from astropy.stats import median_absolute_deviation
from astropy.time import Time
import collections
from gPhoton import gphoton_utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.ticker import MultipleLocator
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import _pickle as pickle
from progressbar import ProgressBar
import random
random.seed()
import seaborn as sns
import time
import WDranker_2
import WDutils

desc="""
WD_Recovery: Procedure for injecting and recovering synthetic optical 
data from WD-1145.
"""

def calculate_jd(galex_time):
    """
    Calculates the Julian date, in the TDB time standard, given a GALEX time.

    :param galex_time: A GALEX timestamp.

    :type galex_time: float

    :returns: float -- The time converted to a Julian date, in the TDB
        time standard.
    """

    if np.isfinite(galex_time):
        # Convert the GALEX timestamp to a Unix timestamp.
        this_unix_time = Time(galex_time + 315964800., format="unix",
                              scale="utc")

        # Convert the Unix timestamp to a Julian date, measured in the
        # TDB standard.
        this_jd_time = this_unix_time.tdb.jd
    else:
        this_jd_time = np.nan

    return this_jd_time

def FindCutoff(percentile):
    assert(os.path.isfile('AllData.csv'))
    df = pd.read_csv('AllData.csv')
    cutoff = np.percentile(df['BestRank'], percentile)
    return cutoff

#Define a class for a galex visit df
class Visit:
    def __init__(self, df, filename, mag):
        self.df = df
        self.timereset = False
        self.df = self.df.reset_index(drop=True)
        self.filename = filename
        self.mag = mag
        self.FUVfilename = filename.replace('NUV', 'FUV')
        self.cEXP()
        self.original_median = self.flux_median()

    def cEXP(self):
        tup = WDranker_2.find_cEXP(self.df)
        self.c_exposure = tup.c_exposure

    #Need this for when we revert relative scales
    def flux_median(self):
        df_reduced = WDutils.df_fullreduce(self.df)
        allflux = df_reduced['flux_bgsub']
        median = np.nanmedian(allflux)
        return median
        

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
            jd = calculate_jd(self.df['t_mean'][i])

            if i == 0:
                tmin = jd

            if i == len(self.df['t_mean'])-1:
                tmax = jd

            newtime = (jd - tmin) * 1440
            self.df.loc[i, 't_mean'] = newtime
        self.timereset = True
        self.tmin = tmin
        self.tmax = tmax

    def FUVexists(self):
        if os.path.isfile('FUV/'+self.FUVfilename):
            return True
        else:
            return False
    
    def FUVmatch(self):
        if self.FUVexists():
            alldataFUV = pd.read_csv('FUV/'+self.FUVfilename)
            alldataFUV = WDutils.df_fullreduce(alldataFUV)
            alldataFUV = WDutils.tmean_correction(alldataFUV)
            FUV_relativetup = WDutils.relativescales_1(alldataFUV)
            alldataFUV_t_mean = FUV_relativetup.t_mean
            alldataFUV_flux = FUV_relativetup.flux
            alldataFUV_flux_err = FUV_relativetup.err

            if self.timereset==False:
                self.reset_time()
            for i in range(len(alldataFUV['t_mean'])):
                jd = calculate_jd(alldataFUV['t_mean'][i])
                alldataFUV.loc[i, 't_mean'] = jd
            FUV_relativetup = WDutils.relativescales_1(alldataFUV)
            alldataFUV_t_mean = FUV_relativetup.t_mean
            alldataFUV_flux = FUV_relativetup.flux
            alldataFUV_flux_err = FUV_relativetup.err

            idx_FUV = np.where( (alldataFUV_t_mean >= self.tmin)
                               &(alldataFUV_t_mean <= self.tmax))[0]
            t_meanFUV = np.array(alldataFUV_t_mean[idx_FUV])
            flux_FUV = np.array(alldataFUV_flux[idx_FUV])
            flux_err_FUV = np.array(alldataFUV_flux_err[idx_FUV])

            t_meanFUV = [ (jd-self.tmin)*1440 for jd in t_meanFUV ]
    
            OutputTup = collections.namedtuple('OutputTup', ['t_mean', 
                                                             'flux', 
                                                             'err'])
            tup = OutputTup(t_meanFUV, flux_FUV, flux_err_FUV)

            return tup
        else:
            return None

    #Inject an optical lc and scale by multiplicative factor
    def inject(self, opticalLC, mf, plot=False):
        if self.timereset==False:
            self.reset_time()

        if self.FUVexists():
            exists=True
            FUVtup = self.FUVmatch()
            self.t_mean_FUV = FUVtup.t_mean
            flux_FUV = FUVtup.flux
            flux_err_FUV = FUVtup.err
        else:
            exists=False

        t_mean = self.df['t_mean']
        #Shift to relative flux scales
        relativetup = WDutils.relativescales_1(self.df)
        flux = relativetup.flux
        flux_err = relativetup.err

        if plot:
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6))
            ax0.errorbar(t_mean, flux, yerr=flux_err, color='red')
            if exists:
                ax0.errorbar(self.t_mean_FUV, flux_FUV, 
                             yerr=flux_err_FUV, color='blue')
            ax0.axhline(np.median(flux), color='black', ls='--')
            ax0.set_title("Original LC")
            ax0 = WDutils.plotparams(ax0)
        
        #Get optical data
        df_optical, ax_optical = selectOptical(opticalLC, 
                                        exposure=self.exposure/60, plot=plot)

        #linear interpolation to match times
        optical_flux = np.interp(t_mean, df_optical['time'], 
                                 df_optical['flux'])

        if exists:
            optical_flux_FUV = np.interp(self.t_mean_FUV, df_optical['time'],
                                         df_optical['flux'])

        #Scale by multiplicative factor
        optical_flux = [ (o - 1)*mf + 1 for o in optical_flux ]
        flux_injected = [ flux[i] * optical_flux[i] 
                              for i in range(len(flux)) ]

        #Put back into df
        self.df.loc[:, 'flux_bgsub'] = flux_injected
        self.df.loc[:, 'flux_bgsub_err'] = flux_err 

        #Do the same for the FUV
        if exists:
            optical_flux_FUV = [ (o -1)*mf + 1 for o in optical_flux_FUV ]
            flux_injected_FUV = [ flux_FUV[i] * optical_flux_FUV[i]
                                    for i in range(len(flux_FUV)) ]
            self.flux_injected_FUV = flux_injected_FUV
            self.flux_err_FUV = flux_err_FUV


        #Remove colored points (flag, expt, sigma clip)
        coloredtup = WDutils.ColoredPoints(self.df)
        droppoints = np.unique(np.concatenate([coloredtup.redpoints, 
                                               coloredtup.bluepoints]))
        self.df = self.df.drop(index=droppoints)
        self.df = self.df.reset_index(drop=True)

        self.t_mean = self.df['t_mean']
        self.flux_injected = self.df['flux_bgsub']
        self.flux_err = self.df['flux_bgsub_err']

        if plot:
            ax1.errorbar(self.t_mean, self.flux_injected, 
                         yerr=self.flux_err, color='red')
            if exists:
                ax1.errorbar(self.t_mean_FUV, self.flux_injected_FUV,
                             yerr=self.flux_err_FUV, color='blue')
            ax1.axhline(np.median(self.flux_injected), color='black', ls='--')
            ax1.set_title("Injected LC")
            ax1 = WDutils.plotparams(ax1)

            plt.show()
    
    #Use the periodogram metric to test if injected signal is recoverable
    def assessrecovery(self):
        exists = self.FUVexists()

        #Exposure metric already computed in init (self.c_exposure)

        #Periodogram Metric
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
        
        '''
        if (not ditherperiod_exists) and (c_periodogram > 0):
            return 1
        else:
            return 0
        '''

        #Welch Stetson Metric
        if exists:
            c_ws = WDranker_2.find_cWS(self.t_mean, self.t_mean_FUV, 
                                       self.flux_injected, 
                                       self.flux_injected_FUV,
                                       self.flux_err, self.flux_err_FUV,
                                       ditherperiod_exists, self.FUVexists())
        else:
            c_ws = WDranker_2.find_cWS(self.t_mean, None,
                                       self.flux_injected, None,
                                       self.flux_err, None,
                                       ditherperiod_exists, self.FUVexists())

        #RMS Metric --- have to 'unscale' the magnitudes
        converted_flux = [ f*self.original_median 
                           for f in self.flux_injected ]
        injectedmags = [ WDutils.flux_to_mag('NUV', f) 
                         for f in converted_flux ]
        sigma_mag = median_absolute_deviation(injectedmags)
        c_magfit = WDranker_2.find_cRMS(self.mag, sigma_mag, 'NUV')

        #Weights:
        w_pgram = 1
        w_expt = .2
        w_WS = .3
        w_magfit = .25

        C = ((w_pgram * c_periodogram) 
            + (w_expt * self.c_exposure) 
            + (w_magfit * c_magfit) 
            + (w_WS * c_ws))

        #cutoff = FindCutoff(95) 
        cutoff = .665 #Don't waste time loading in alldata
        #print("Rank --- ", C, "Cutoff --- ", cutoff)

        if C > cutoff:
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
def selectLC(binvalue, binsize, mag_array, path_array):
    #First need to find indicies for bin
    binupper = binvalue + binsize 
    idx_bin = np.where( (mag_array >= binvalue)
                       &(mag_array < binupper) )[0]
    filename = np.random.choice(path_array[idx_bin])
    
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
        visit = Visit(df, filename, source_mag)
        if visit.good_df() == True: 
            if visit.existingperiods() == False:
                visit_list.append(visit)

    #If there were no good visits, pick a new source
    if len(visit_list) == 0:
        print(filename, "no sources in visit list")
        visit, source_mag = selectLC(binvalue, binsize, mag_array, path_array)
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
        #plt.show()
        return df_optical, ax
    else:
        return df_optical, None

#Full injection procedure for single source
def main(mag, bs, mag_array, path_array, opticalLC, fname, verbose):
    #Select a visit
    visit, source_mag = selectLC(mag, bs, mag_array, path_array)

    #Select a multiplicative factor
    mf = random.random() * 2
    mf = round(mf, 3) 
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

    if verbose:
        print(outputstr)
    return tup

def wrapper(mag_array, path_array, opticalLC, 
            iterations, ml, mu, bs, p, fname, verbose):
    #Set up magnitude bins
    minbin=ml
    maxbin=mu
    bins = np.arange(minbin, maxbin+bs, bs)
    bins = np.array([ round(b, 1) for b in bins ])

    #Create mp pool and iterate
    pool = mp.Pool(processes = p)
    jobs=[]
    for i in range(iterations):
        for mag in bins:
            job = pool.apply_async(main, args=(mag, bs, mag_array, 
                                               path_array, opticalLC, 
                                               fname, verbose,))
            jobs.append(job)
    for job in jobs:
        job.get()

#Iterate through all visits of a source
def testfunction(filename, output):
    usecols = ['t0', 't1', 't_mean',
               'mag_bgsub',
               'cps_bgsub', 'cps_bgsub_err', 'counts',
               'flux_bgsub', 'flux_bgsub_err',
               'detrad', 'flags', 'exptime']
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
        visit = Visit(df, filename, source_mag)
        if visit.good_df() == True: 
            if visit.existingperiods() == False:
                visit_list.append(visit)

    #If there were no good visits, pick a new source
    if len(visit_list) == 0:
        outputstr = filename + " no sources in visit list"
        if output is not None:
            fname = output
            os.system("echo {0} >> {1}".format(outputstr, fname))
    else:
        for visit in visit_list:
            mf = random.random() * 2
            mf = round(mf, 3) 
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
            if output is not None:
                fname = output
                os.system("echo {0} >> {1}".format(outputstr, fname))

#Multiprocessing of testfunction
def testfunction_wrapper(path_array, output, p):
    #Create mp pool and iterate
    pool = mp.Pool(processes = p)
    jobs=[]
    for filename in path_array:
        job = pool.apply_async(testfunction, args=(filename, output,))
        jobs.append(job)
    for job in jobs:
        job.get()

def plot(fname, ml, mu, bs, magarray):
    assert(os.path.isfile(fname))
    with open(fname) as f:
        lines = f.readlines()
    mag = [[float(x.strip()) for x in line.split(',')][0] for line in lines]
    mf = [[float(x.strip()) for x in line.split(',')][1] for line in lines]
    out = [[float(x.strip()) for x in line.split(',')][2] for line in lines]

    minbin=14
    maxbin=22
    bs = .1
    magbins = np.arange(minbin, maxbin+bs, bs)
    magbins = np.array([ round(b, 1) for b in magbins ])

    mfbins = np.arange(0, 2, .1)
    mfbins = np.array([ round(b, 1) for b in mfbins ])

    array2d = np.array([magbins, mfbins])
    totalarray = np.zeros((len(magbins), len(mfbins)))
    recoveryarray = np.zeros((len(magbins), len(mfbins)))
    resultarray = np.zeros((len(magbins), len(mfbins)))
    for i in range(len(mag)):
        magval = mag[i]
        mfval = mf[i]
        result = out[i]
        magidx = np.where(array2d[0] <= magval)[0].max()
        mfidx = np.where(array2d[1] <= mfval)[0].max()
        totalarray[magidx, mfidx] += 1
        recoveryarray[magidx, mfidx] += result
    for x in range(totalarray.shape[0]):
        for y in range(totalarray.shape[1]):
            if totalarray[x, y] == 0:
                resultarray[x, y] = 0
            else:
                resultarray[x,y] = (recoveryarray[x,y] / 
                                    totalarray[x,y]) * 100

    fig = plt.figure(figsize=(12, 5))
    gs1 = gs.GridSpec(3, 7)
    plt.subplots_adjust(top=.98, right=.98, bottom = 0.25, 
                        hspace=0, wspace=0)

    plt.subplot2grid((3,7),(1,0), colspan=6, rowspan=2)
    ax0 = plt.gca()
    cax = ax0.imshow(np.flip(resultarray,1).T, cmap='viridis', 
                     extent=(14,22,0,2), aspect='auto')
    ax0.set_xlabel(r'$GALEX$'+' NUV (mag)', fontsize=25)
    ax0.set_ylabel('Scale Factor', fontsize=25)
    ax0.set_xticks([x for x in np.arange(14, 22, 1)])
    ax0 = WDutils.plotparams(ax0)
    ax0.xaxis.get_major_ticks()[0].set_visible(False)
    p0 = ax0.get_position().get_points().flatten()

    plt.subplot2grid((3,7),(0,0),colspan=6, rowspan=1)
    axM = plt.gca()
    axM.bar(magbins+.05, resultarray.T.mean(axis=0), 
            width=.1,color='#62CA5F', alpha=.8, edgecolor='black')
    axM.set_xlim(xmin=14, xmax=22)
    axM.xaxis.set_ticklabels([])
    axM = WDutils.plotparams(axM)
    axM.yaxis.tick_right()
    axM.yaxis.get_major_ticks()[0].set_visible(False)
    axM.spines['bottom'].set_linewidth(2)
    axM.set_ylim(ymax=axM.get_ylim()[1]+5)
    pM = axM.get_position().get_points().flatten()

    plt.subplot2grid((3,7),(1,6),colspan=1, rowspan=2)
    axS = plt.gca()
    axS.barh(mfbins+.05,resultarray.mean(axis=0), 
             height=.1,color='#62CA5F', alpha=.8, edgecolor='black')
    axS.set_ylim(ymin=0, ymax=2)
    axS.yaxis.set_ticklabels([])
    axS = WDutils.plotparams(axS)
    axS.xaxis.tick_top()
    axS.xaxis.get_major_ticks()[0].set_visible(False)
    axS.spines['left'].set_linewidth(2)
    pS = axS.get_position().get_points().flatten()

    cbaxes = fig.add_axes([p0[0], .05, p0[2]-p0[0], .05])
    cb = plt.colorbar(cax, cbaxes, orientation='horizontal')

    fig.text(.92, .94, "%\nRecovered", 
             ha='center', va='center', color='black', fontsize=18)
    fig.savefig('RecoveryPlot.pdf')


    
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
    parser.add_argument("--v", help='Verbose', 
                        default=False, action='store_true')
    parser.add_argument("--test", help="Use test function wrapper",
                        default=False, action='store_true')
    parser.add_argument("--plot", help="Make plot with input file name", 
                        default=None, type=str)
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

    if args.test:
        testfunction_wrapper(path_array, args.output, args.p)
    elif args.plot is not None:
        plot(args.plot, args.ml, args.mu, args.bs, mag_array)
    else:
        wrapper(mag_array, path_array, opticalLC, args.iter, args.ml,
                args.mu, args.bs, args.p, args.output, args.v)
