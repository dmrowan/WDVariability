#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import argparse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import LombScargle
from astropy.stats import median_absolute_deviation
from astropy.time import Time
import collections
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patheffects import withStroke
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import _pickle as pickle
from progressbar import ProgressBar
import random
random.seed()
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
        self.exposure = tup.exposure

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
    def inject(self, opticalLC, mf, plot=False, center=None):
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
                                        exposure=self.exposure/60, 
                                        plot=plot, center=center)

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

            #plt.show()
            return ax1
    
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
    bigcatalog = pd.read_csv("MainCatalog.csv")
    k2catalog = pd.read_csv("K2.csv")
    k2ra = []
    k2dec = []
    for i in range(len(k2catalog['RA'])):
        ra_c = k2catalog['RA'][i]
        dec_c = k2catalog['Dec'][i]
        s= ra_c + " " + dec_c
        c = SkyCoord(s, unit=(u.hourangle, u.degree))
        k2ra.append(c.ra.deg)
        k2dec.append(c.dec.deg)
    kepler_catalog = SkyCoord(ra=k2ra*u.degree,dec=k2dec*u.degree)
    #Iterate through NUV files in cwd
    pbar = ProgressBar()
    for filename in pbar(os.listdir(os.getcwd())):
        if filename.endswith('.csv') and 'NUV' in filename:
            source = filename[:-8]
            our_idx = WDutils.catalog_match(source, bigcatalog)
            if len(our_idx) == 0:
                print("No catalog match for ", filename)
                continue
            else:
                our_idx = our_idx[0]
                our_ra = bigcatalog['ra'][our_idx]
                our_dec = bigcatalog['dec'][our_idx]
                our_c = SkyCoord(ra=our_ra*u.deg, dec=our_dec*u.deg)
                kidx, d2d, d3d = our_c.match_to_catalog_sky(kepler_catalog)
                if d2d < 5*u.arcsec:
                    kidx = int(kidx)
                    #print("Our RA", our_ra)
                    #print("Our DEC", our_dec)
                    #print("Kepler RA",k2catalog['RA'][kidx])
                    #print("Kepler DEC",k2catalog['Dec'][kidx])
                    continue
                else:
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
                            if ((tflagged > (exposure / 4)) 
                                    or (exposure < 500)):
                                continue
                            else:
                                mag_list.append(mag)
                                path_list.append(filename)
                                break
                    continue
    median = np.median(mag_list)
    print("Median magnitude --- ", median)

    #Magnitude Histogram
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(12, 12))
        bins = np.arange(min(mag_list), max(mag_list)+.1, .1)
        ax.hist(mag_list, bins=bins, color='xkcd:red', linewidth=1.2, 
                edgecolor='black')
        ax = WDutils.plotparams(ax)

        fig.savefig("/home/dmrowan/WhiteDwarfs/NUVmaghist.pdf")

    #Save as pickle
    
    print("saving pickle")
    print(len(mag_list), len(path_list))
    mag_array = np.array(mag_list)
    path_array = np.array(path_list)
    tup = (mag_array,path_array)
    with open('MagList2.pickle', 'wb') as handle:
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
        print(filename)
        visit = random.choice(visit_list)
        print(np.where(np.array(visit_list) == visit)[0])
        return visit, source_mag


#Select random observation and time chunk
def selectOptical(opticalLC, plot=False, exposure=30, center=None):
    exphalf = exposure/2
    maxtime = max(opticalLC['time'])
    #Find idx range corresponding to exposure
    idx_low = np.where(opticalLC['time'] < exphalf)[0][-1]
    idx_high = np.where(opticalLC['time'] > maxtime-exphalf)[0][0]
    idx_center = np.arange(idx_low, idx_high, 1)
    if center is None:
        time_center = np.random.choice(opticalLC['time'][idx_center])
    else:
        time_center = center
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

def gen_resultarray(fname):
    assert(os.path.isfile(fname))
    #Parse results
    with open(fname) as f:
        lines = f.readlines()
    mag = [[float(x.strip()) for x in line.split(',')][0] for line in lines]
    mf = [[float(x.strip()) for x in line.split(',')][1] for line in lines]
    out = [[float(x.strip()) for x in line.split(',')][2] for line in lines]

    #Define our axes
    minbin=14
    maxbin=22
    bs = .1
    magbins = np.arange(minbin, maxbin+bs, bs)
    magbins = np.array([ round(b, 1) for b in magbins ])

    mfbins = np.arange(0, 2, .1)
    mfbins = np.array([ round(b, 1) for b in mfbins ])

    #Fill 2d numpy arrays 
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
    #Use recovery/total to calculate result
    for x in range(totalarray.shape[0]):
        for y in range(totalarray.shape[1]):
            if totalarray[x, y] == 0:
                resultarray[x, y] = 0
            else:
                resultarray[x,y] = (recoveryarray[x,y] / 
                                    totalarray[x,y]) * 100

    with open('resultarray.pickle', 'wb') as p:
        pickle.dump(resultarray, p)

    return resultarray

#Recovery plot with mf cuts in additional subplot
def ProjectionPlot(resultarray):
    #Define our axes
    minbin=14
    maxbin=22
    bs = .1
    magbins = np.arange(minbin, maxbin+bs, bs)
    magbins = np.array([ round(b, 1) for b in magbins ])

    mfbins = np.arange(0, 2, .1)
    mfbins = np.array([ round(b, 1) for b in mfbins ])

    #Gridspec plot
    fig, outer_ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.subplots_adjust(bottom=.15)
    #gs_outer = gs.GridSpec(1, 1)
    #outer_ax = plt.subplot(gs_outer[0])
    #Subplot for imshow
    gs_inner = gs.GridSpecFromSubplotSpec(2,1, subplot_spec=outer_ax,
                                          hspace=0, height_ratios=[1,1.25])
    ax0 = plt.subplot(gs_inner[1])

    #Use np.flip and T to set up axes correctly
    im = ax0.imshow(np.flip(resultarray,1).T, cmap='viridis', 
                    extent=(14,22,0,2), aspect='auto')
    ax0.set_xlabel(r'$GALEX$'+' NUV (mag)', fontsize=25)
    ax0.set_ylabel('Scale Factor', fontsize=25)
    ax0.set_xticks([x for x in np.arange(14, 22, 1)])
    ax0 = WDutils.plotparams(ax0)
    ax0.xaxis.get_major_ticks()[0].set_visible(False)

    #Subplot for magnitude bar
    axM = plt.subplot(gs_inner[0])
    mf_values = [.5, 1, 1.5]
    current_z = 3
    colors = ['xkcd:azure', 'xkcd:blue', 'xkcd:darkblue']
    for i in range(len(mf_values)):
        idx = np.where(mfbins == mf_values[i])[0]
        slicevalues = resultarray.T[idx]
        
        axM.bar(magbins+.05, slicevalues[0], 
               label=r'$s='+str(mf_values[i])+r'$',
               width=.1, edgecolor='black', zorder=current_z, 
               color=colors[i])

        current_z -= 1

    axM.legend(loc=1)
    axM.set_xlim(xmin=14, xmax=22)
    axM.xaxis.set_ticklabels([])
    axM = WDutils.plotparams(axM)
    #axM.yaxis.get_major_ticks()[0].set_visible(False)
    axM.spines['bottom'].set_linewidth(2)
    axM.set_ylim(ymax=axM.get_ylim()[1]+5)
    axM.set_ylabel("Percent\nRecovered", fontsize=20)
    
    axcb = fig.colorbar(im, ax=outer_ax.ravel().tolist(), pad=.04, aspect=30)

    fig.savefig('ProjectionPlot.pdf')

#Basic recovery plot with single subplot
def RecoveryPlot(resultarray):

    #Define our bin arrays
    minbin=14
    maxbin=22
    bs = .1
    magbins = np.arange(minbin, maxbin+bs, bs)
    magbins = np.array([ round(b, 1) for b in magbins ])

    mfbins = np.arange(0, 2, .1)
    mfbins = np.array([ round(b, 1) for b in mfbins ])

    #Gridspec plot
    fig, ax0 = plt.subplots(1, 1, figsize=(12,4))

    #Subplot for array
    ax0 = plt.gca()
    #Use np.flip and T to set up axes correctly
    im = ax0.imshow(np.flip(resultarray,1).T, cmap='viridis', 
                     extent=(14,22,0,2), aspect='auto')
    ax0.set_xlabel(r'$GALEX$'+' NUV (mag)', fontsize=25)
    ax0.set_ylabel('Scale Factor', fontsize=25)
    ax0.set_xticks([x for x in np.arange(14, 22, 1)])
    ax0 = WDutils.plotparams(ax0)
    ax0.xaxis.get_major_ticks()[0].set_visible(False)


    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=.1)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.set_ylabel('Percent Recovered', fontsize=25)
    plt.subplots_adjust(bottom=.175, top=.98)
    fig.savefig('RecoveryPlot.pdf')

#Comparison of different recovery scenarios
def ComparisonPlot(fname, opticalLC):

    assert(os.path.isfile(fname))
    with open(fname) as f:
        lines = f.readlines()
    brightsource = lines[0][:-1] #visit i=0
    middlesource = lines[1][:-1] #visit i=2
    faintsource = lines[2][:-1] #visit i=0

    visit_list = []
    mag_list = []
    for sourcefile, i in zip([brightsource, middlesource, faintsource],
                           [0,2,0]):
        print(sourcefile)
        assert(os.path.isfile(sourcefile))
        usecols = ['t0', 't1', 't_mean',
                   'mag_bgsub',
                   'cps_bgsub', 'cps_bgsub_err', 'counts',
                   'flux_bgsub', 'flux_bgsub_err',
                   'detrad', 'flags', 'exptime']
        alldata = pd.read_csv(sourcefile, usecols=usecols)
        #Data reduction and time correction
        alldata = WDutils.df_reduce(alldata)
        alldata = WDutils.tmean_correction(alldata)
        #Split into visits 
        data = WDutils.dfsplit(alldata, 100)
        source_mag = round(np.nanmedian(alldata['mag_bgsub']),5)
        mag_list.append(round(source_mag,1))

        good_list = []
        for j in range(len(data)):
            df = data[j]
            vtemp = Visit(df, sourcefile, source_mag)
            if vtemp.good_df() == True: 
                if vtemp.existingperiods() == False:
                    good_list.append(j)
        
        #Select df
        df = data[good_list[i]]
        v0 = Visit(df, sourcefile, source_mag) #mf=1 recovered
        v1 = Visit(df, sourcefile, source_mag) #mf=1 nonrecovered
        v2 = Visit(df, sourcefile, source_mag) #mf=.25 recovered
        v3 = Visit(df, sourcefile, source_mag) #mf=.25 nonrecovered
        visit_list.append([v0, v1, v2, v3])

    df_optical, unused = selectOptical(opticalLC, plot=False, 
                                       exposure=30, center=185)

    mflist = [1,1,.25,.25]

    myeffectw = withStroke(foreground='black', linewidth=2)
    txtkwargsw = dict(path_effects=[myeffectw])
    afont = {'fontname':'Keraleeyam'}

    fig = plt.figure(figsize=(12, 22))
    gs_outer = gs.GridSpec(4, 1, height_ratios=[1,3,3,3], hspace=.15)
    gs_optical = gs.GridSpecFromSubplotSpec(1,1, subplot_spec=gs_outer[0])
    axOptical = plt.subplot(gs_optical[0])
    axOptical.scatter(df_optical['time'], df_optical['flux'], 
                      color='xkcd:violet', marker='.', s=20)
    axOptical = WDutils.plotparams(axOptical)

    gs1 = gs.GridSpecFromSubplotSpec(2,2, subplot_spec=gs_outer[1], 
            hspace=0, wspace=0)
    gs2 = gs.GridSpecFromSubplotSpec(2,2, subplot_spec=gs_outer[2],
            hspace=0, wspace=0)
    gs3 = gs.GridSpecFromSubplotSpec(2,2, subplot_spec=gs_outer[3],
            hspace=0, wspace=0)

    plt.subplots_adjust(top=.98, right=.98, bottom=.05)

    gs_list = [gs1, gs2, gs3]
    for i in range(len(visit_list)):
        gs_c = gs_list[i]
        use_ylabel = True
        for ii in range(len(visit_list[i])):
            mf = mflist[ii]
            visit = visit_list[i][ii]
            visit.inject(opticalLC, mf, center=185)
            
            ax_c = plt.subplot(gs_c[ii])
            ax_c.errorbar(visit.t_mean, visit.flux_injected, 
                         yerr=visit.flux_err, color='red',
                         marker='.', ls='', alpha=.5, ms=10)
            if visit.FUVexists():
                ax_c.errorbar(visit.t_mean_FUV, visit.flux_injected_FUV,
                             yerr=visit.flux_err_FUV, color='blue',
                             marker='.', ls='', alpha=.5, ms=10)
            ax_c = WDutils.plotparams(ax_c)

            if use_ylabel:
                use_ylabel = False
            else:
                ax_c.yaxis.set_ticklabels([])
                use_ylabel = True

            if ii in [0,1]:
                ax_c.xaxis.set_ticklabels([])

            ax_c.text(.1, .9, str(mag_list[i]), 
                      transform=ax_c.transAxes,
                      color='gray', fontsize=20, 
                      ha='center', va='top', 
                      **afont, **txtkwargsw,
                      bbox=dict(boxstyle='square', fc='w', ec='none',
                          alpha=.3))
            if ii in [0, 2]:
                mfcolor='green'
                if ii==0:
                    mftext=r'$s=1$'
                else:
                    mftext=r'$s=.25$'
            else:
                mfcolor='red'
                if ii==1:
                    mftext=r'$s=1$'
                else:
                    mftext=r'$s=.25$'
            ax_c.text(.9, .9, mftext, transform=ax_c.transAxes,
                      color=mfcolor, fontsize=18,
                      ha='center', va='top',
                      **afont, **txtkwargsw,
                      bbox=dict(boxstyle='square', fc='w', ec='none',
                          alpha=.3))

            if ii == 0:
                print(ax_c.get_position())

    plt.subplots_adjust(top=.98, right=.98, bottom=.05)
    for y in [.03, .315, .595, .88]:
        fig.text(.56, y, 'Time Elapsed (min)', 
                 va='center', ha='center', fontsize=20)
    for y in [.18, .45, .733, .94]:
        fig.text(.05, y, 'Normalized Flux', va='center', 
                 rotation='vertical', fontsize=20, ha='center')
    fig.savefig("TestCompare.pdf")

def SlicePlot(resultarray):
    #Define our axes
    minbin=14
    maxbin=22
    bs = .1
    magbins = np.arange(minbin, maxbin+bs, bs)
    magbins = np.array([ round(b, 1) for b in magbins ])

    mfbins = np.arange(0, 2, .1)
    mfbins = np.array([ round(b, 1) for b in mfbins ])

    #Subplot for magnitude bar
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 3))
    #mf_values = [.5, 1, 1.5]
    mf_values = [.5, 1, 1.5]
    #z_values = [3, 2, 1]
    z_values = [1, .5, 0]
    colors = ['xkcd:azure', 'xkcd:blue', 'xkcd:darkblue']
    #Create three separate bar grapsh
    for i in range(len(mf_values)):
        idx = np.where(mfbins == mf_values[i])[0]
        slicevalues = resultarray.T[idx]
        
        ax.bar(magbins+.05, slicevalues[0], 
               label=r'$s='+str(mf_values[i])+r'$',
               width=.1, edgecolor='black', zorder=z_values[i],
               color=colors[i])

    
    ax.set_xlabel(r'$GALEX$ NUV (mag)', fontsize=20)
    ax.set_ylabel('Percent Recovered', fontsize = 20)

    ax.set_xlim(14, 22)
    ax.legend(fontsize=15, loc=1,
              edgecolor='black', framealpha=.9, markerscale=.2)

    ax = WDutils.plotparams(ax)
    ax.tick_params(axis='both', which='both', zorder=2)
    ax.set_yticks([25, 50, 75, 100])

    plt.subplots_adjust(right=.98, bottom=.2)
    fig.savefig('SlicePlot.pdf')

#Find what fraction of of the optical light curve has detectable transits
def DetectableTransits(opticalLC, threshold):
    iterations = 10000
    if not isinstance(threshold, list):
        threshold = [ threshold ]
    for i in range(len(threshold)):
        if threshold[i] > 1:
            threshold[i] = threshold[i] / 100
    results = []
    for t in threshold:
        print("Threshold: ", t)
        counter = 0
        pbar = ProgressBar()
        for i in pbar(range(iterations)):
            df_optical, ax = selectOptical(opticalLC, plot=False, exposure=30)
            if min(df_optical['flux']) <= t:
                counter += 1
            else:
                continue
        fraction = counter / iterations
        results.append(fraction)
    d = dict(zip(threshold, results))
    return d

def maglist_hist(mag_array, path_array):
    #Generate Figure
    fig = plt.figure(figsize=(12, 6))
    gs1 = gs.GridSpec(2, 1, hspace=0)
    fig.text(.5, .05, r'$GALEX$ NUV (mag)', va='center', ha='center',
             fontsize=30)

    ax0 = plt.subplot2grid((2, 1), (0, 0), colspan=1, rowspan=1)
    bins = np.arange(14, 21.1, .1)
    ax0.hist(mag_array, bins=bins, color='xkcd:red', linewidth=1.2, 
            edgecolor='black')
    ax0.set_xlim(14, 21)
    ax0 = WDutils.plotparams(ax0)

    if os.path.isfile('visitarray.pickle'):
        with open('visitarray.pickle', 'rb') as p:
            n_visits = pickle.load(p)
    else:
        n_visits = np.zeros(len(bins))
        pbar = ProgressBar()
        for filename in pbar(path_array):
            print(filename)
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

            idx_bins = np.where(bins <= source_mag)[0]
            if len(idx_bins) != 0:
                idx_bins = idx_bins.max()
                n_visits[idx_bins] += len(visit_list)

        with open('visitarray.pickle', 'wb') as p:
            pickle.dump(n_visits, p)
    ax1 = plt.subplot2grid((2, 1), (1, 0), colspan=1, rowspan=1)
    ax1.bar(bins, n_visits, width = .1, edgecolor='black', 
            color='xkcd:darkblue', align='edge')
    ax1.set_xlim(14, 21)   
    ax1 = WDutils.plotparams(ax1)

    fig.savefig('MagnitudeHistograms.pdf')


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--iter", help="Number of iterations to run", 
                        type=int, default=100)
    parser.add_argument("--ml", help="Lower mag range", 
                        type=float, default=14)
    parser.add_argument("--mu", help="Upper mag range", 
                        type=float, default=22)
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
    parser.add_argument("--plot", help="3 plotting options",
                        default=False, action='store_true')
    parser.add_argument("--dt", help="Find detectable transit percentage",
                        default=False, action='store_true')
    parser.add_argument("--gen_array", help="Generate result array pickle",
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
    with open('MagList2.pickle', 'rb') as p2:
        MagList = pickle.load(p2)
    if os.path.isfile('resultarray.pickle'):
        with open('resultarray.pickle', 'rb') as p3:
            resultarray = pickle.load(p3)
    
    mag_array = MagList[0]
    path_array = MagList[1]

    
    
    if args.test:
        testfunction_wrapper(path_array, args.output, args.p)
    elif args.dt:
        result = DetectableTransits(opticalLC, [.9, .75, .5])
        print(result)
    elif args.plot:
        selection = input("(1) Projection Plot \n"
                          +"(2) Recovery Plot \n"
                          +"(3) Comparison Plot \n"
                          +"(4) Slice Plot \n"
                          +"(5) Magnitude Hist \n"
                          +"---Select Plot Type---")
        if selection == '1':
            ProjectionPlot(resultarray)
        elif selection == '2':
            RecoveryPlot(resultarray)
        elif selection == '3':
            ComparisonPlot(resultarray, opticalLC)
        elif selection == '4':
            SlicePlot(resultarray)
        elif selection == '5':
            maglist_hist(mag_array, path_array)

        else:
            print("Invalid selection")
    elif args.gen_array is not None:
        gen_resultarray(args.gen_array)
    else:
        wrapper(mag_array, path_array, opticalLC, args.iter, args.ml,
                args.mu, args.bs, args.p, args.output, args.v)
   
   

