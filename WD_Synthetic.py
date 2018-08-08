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

desc="""
WD_Synthetic: Define a class synthetic planets
"""
class SyntheticVisit:
    
    def __init__(self, visit):
        #Visit object
        self.initial = visit
        self.source = self.initial.source

    def inject(self, center=15, width=3, amp=100, plot=False):
        if self.initial.good_df == False:
            return
        
        center = max(self.initial.t_mean) - min(self.initial.t_mean)
        mu = center
        sigma = width / (2*np.sqrt(2*np.log(2)))
        xarray = self.initial.t_mean
        gaussian = [ (1/np.sqrt(2*np.pi*sigma**2))
                     *np.exp((-(x-mu)**2) / (2*sigma**2))
                     for x in xarray ] 
        invertedgaussian = [ -amp*g for g in gaussian ]

        #Injected flux bgsub
        flux_bgsub_inj = []
        for ii in range(len(self.initial.flux_bgsub)):
            newvalue = (list(self.initial.flux_bgsub)[ii] 
                        + invertedgaussian[ii])
            err = self.initial.flux_bgsub_err[ii]
            newvalue += np.random.normal(newvalue, err)
            flux_bgsub_inj.append(newvalue)

        self.flux_bgsub_inj = flux_bgsub_inj

        if plot:
            bandcolors = {'NUV':'red', 'FUV':'blue'}
            band = self.initial.AllLC.band
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(20,16))
            #ax1.errorbar(jd_t_mean, flux_bgsub, yerr=flux_bgsub_err, 
                        #color=bandcolors[self.band],
                        #marker='.', ls='', zorder=4, label=self.band)

            ax1.plot(xarray,invertedgaussian, color='black')

            ax1.errorbar(self.initial.t_mean, 
                         flux_bgsub_inj, 
                         yerr=self.initial.flux_bgsub_err,
                         color=bandcolors[band],
                         marker='.', ls='', zorder=4)
            ax1 = WDutils.plotparams(ax1)

            time_seconds = self.initial.t_mean * 60
            ls = LombScargle(time_seconds, flux_bgsub_inj)
            freq, amp = ls.autopower(nyquist_factor=1)

            detrad = self.initial.detrad
            ls_detrad = LombScargle(time_seconds, detrad)
            freq_detrad, amp_detrad = ls_detrad.autopower(nyquist_factor=1)

            exptime = self.initial.exptime
            ls_expt = LombScargle(time_seconds, exptime)
            freq_expt, amp_expt = ls_expt.autopower(nyquist_factor=1)
            
            pgram_tup = WDranker_2.find_cPGRAM(ls, amp_detrad, 
                                               exposure=self.initial.exposure)
            c_periodogram = pgram_tup.c

            print(c_periodogram)
            ax2.plot(freq, amp, 'g-', label='Data')
            ax2.plot(freq_detrad, amp_detrad, 'r-', label='Detrad', alpha=.25)
            ax2.plot(freq_expt, amp_expt, 'b-', label='Exposure', alpha=.25)
            ax2 = WDutils.plotparams(ax2)
            ax2.set_xlim(0, np.max(freq))
            ax2.set_ylim(0, np.max(amp)* 2)
            top5amp_detrad = heapq.nlargest(5, amp_detrad)
            bad_detrad = pgram_tup.bad_detrad
            if any(np.isnan(x) for x in top5amp_detrad):
                print("No detrad peaks for exposure group" + str(df_number))
            else:
                for tup in bad_detrad:
                    ax2.axvspan(tup[0], tup[1], alpha=.1, color='black')

            for level in [.05]:
                ax2.axhline(level, color='black', alpha=.5,
                            ls='--', label='FAP: .05')
            ax2.axhline(.25, color='black', alpha=.5,
                        ls=':', label=' FAP: .25')
            ax2.legend()

            plt.show()

    def assessrecovery(self):
        time_seconds = self.initial.t_mean * 60
        ls = LombScargle(time_seconds, self.flux_bgsub_inj)
        freq, amp = ls.autopower(nyquist_factor=1)

        detrad = self.initial.detrad
        ls_detrad = LombScargle(time_seconds, detrad)
        freq_detrad, amp_detrad = ls_detrad.autopower(nyquist_factor=1)

        pgram_tup = WDranker_2.find_cPGRAM(ls, amp_detrad, 
                                           exposure=self.initial.exposure)
        c_periodogram = pgram_tup.c
        ditherperiod_exists = pgram_tup.ditherperiod_exists
        if (not ditherperiod_exists) and (c_periodogram > 0):
            return True
        else:
            return False

    def detectabilitylimit(self, iterations=1000, width=3):
        sigma = width / (2*np.sqrt(2*np.log(2)))      
        max_amp = np.int(100*np.sqrt(2*np.pi*sigma**2))
        self.namp = 500
        self.iterations = iterations
        amp_list = np.linspace(0, max_amp, self.namp)
        amp_list = np.flip(amp_list, 0)
        pbar = ProgressBar()
        for amp in pbar(amp_list):
            recovery_list = []
            for i in range(self.iterations):
                self.inject(amp=amp, plot=False)
                recovered = self.assessrecovery()
                recovery_list.append(recovered)
            if sum(recovery_list) < self.iterations*.95:
                print(self.source,"Limit is ", amp)
                self.breaklimit = amp
                break

        OutputTup = collections.namedtuple('OutputTup', ['amplimit',
                                                         'gmag'])

        tup = OutputTup(self.breaklimit, self.initial.AllLC.gmag)
        return(tup)

    def savepickle(self):
        fname = "pickles/" + self.source + ".pickle"
        with open(fname, 'wb') as handle:
            pickle.dump(self, handle)

            
class AllLC:
    def __init__(self, csvname):
        assert(os.path.isfile(csvname))

        #Find source name from csvpath
        csvpath = csvname
        for i in range(len(csvpath)):
            character = csvpath[i]
            if character == 'c':
                endidx=i-5
                break
        self.source = csvpath[0:endidx]

        #Grab the band (also checks we have source csv)
        if csvpath[-7] == 'N':
            self.band = 'NUV'
            self.band_other = 'FUV'
        elif csvpath[-7] == 'F':
            self.band = 'FUV'
            self.band_other = 'NUV'
        else:
            print("Not source csv, skipping")
            return
        assert(self.band is not None)

        self.alldata = pd.read_csv(csvpath)

        #Reduction step
        self.alldata = WDutils.df_fullreduce(self.alldata)
        self.alldata = WDutils.tmean_correction(self.alldata)

        #Split step
        data = WDutils.dfsplit(self.alldata, 100)
        self.df_list = data

        if self.band == 'NUV':
            csvpath_other = csvpath.replace('NUV', 'FUV')
        else:
            csvpath_other = csvpath.replace('FUV', 'NUV')

        if os.path.isfile(csvpath_other):
            self.other_band_exists = True
            self.alldata_other = pd.read_csv(csvpath_other)
        else:
            self.other_band_exists = False

        #Catalog query
        catalogpath = ("/home/dmrowan/WhiteDwarfs/"
                       +"Catalogs/MainCatalog_reduced_simbad_asassn.csv")

        assert(os.path.isfile(catalogpath))
        bigcatalog = pd.read_csv(catalogpath)
        bigcatalog_idx = WDutils.catalog_match(self.source, bigcatalog)
        if len(bigcatalog_idx) == 0:
            print(source, "Not in catalog")
        else:
            bigcatalog_idx = bigcatalog_idx[0]
            self.gmag = bigcatalog['gaia_g_mean_mag'][bigcatalog_idx]

    def itervisits(self):
        visit_list = []
        amplimit_list = []
        for i in range(len(self.df_list)):
            df_number = i + 1
            visitobject = visit(self, df_number)
            if visitobject.good_df() == True:
                visitobject.df_params()
                if visitobject.existingperiods() == False:
                    synthvisit = SyntheticVisit(visitobject)
                    dltup = synthvisit.detectabilitylimit()
                    amplimit_list.append(dltup.amplimit)
                    visit_list.append(synthvisit)
        if len(amplimit_list) > 0:
            idx_best = np.where(
                    np.array(amplimit_list) == min(amplimit_list))[0][0]
            best_visit = visit_list[idx_best]
            best_visit.savepickle()
                    
            


class visit:
    def __init__(self, AllLC, df_number):
        #AllLC is AllLC object
        self.df = AllLC.df_list[df_number - 1]
        self.df_number = df_number
        self.AllLC = AllLC
        self.source = self.AllLC.source


    #Return boolean value representing adeqaute data
    def good_df(self):
        if len(self.df['t1']) == 0:
            return False
        else:
            exposuretup = WDranker_2.find_cEXP(self.df)
            exposure = exposuretup.exposure
            tflagged = exposuretup.t_flagged
            if (tflagged > (exposure / 4)) or (exposure < 500):
                return False
            else:
                return True

    #Return parameters of the given df_number (visit)
    def df_params(self):
        good_df = self.good_df
        if len(self.df['t1']) == 0:
            return None

        exposuretup = WDranker_2.find_cEXP(self.df)
        self.exposure = exposuretup.exposure

        firsttime_mean = self.df['t_mean'][self.df.index[0]]
        self.df['t_mean'] = self.df['t_mean'] - firsttime_mean

        #Find exposure (part of c_exposure calculation)
        exposuretup = WDranker_2.find_cEXP(self.df)
        firsttime = exposuretup.firsttime
        lasttime = exposuretup.lasttime
        self.exposure = exposuretup.exposure
        self.c_exposure = exposuretup.c_exposure

        self.df = self.df.reset_index(drop=True)

        for i in range(len(self.df['t_mean'])):
            jd = gphoton_utils.calculate_jd(self.df['t_mean'][i]
                                            +firsttime_mean)
            if i == 0:
                tmin = jd

            newtime = (jd - tmin) * 1440
            self.df.loc[i, 't_mean'] = newtime

        #Shift to relative scales
        #Corrections for relative scales
        relativetup = WDutils.relativescales(self.df)
        self.t_mean = relativetup.t_mean
        self.flux_bgsub = relativetup.flux
        self.flux_bgsub_err = relativetup.err

        self.detrad = self.df['detrad']
        self.exptime = self.df['exptime']
        
        """
        OutputTup = collections.namedtuple('OutputTup', ['good_df',
                                                         'exposure',
                                                         't_mean',
                                                         'flux_bgsub',
                                                         'flux_bgsub_err'])
        tup = OutputTup(self.good_df, self.exposure, self.t_mean, 
                        self.flux_bgsub, self.flux_bgsub_err)
        return tup
        """
    #Check if we already have some period (skip these visits)
    def existingperiods(self):
        time_seconds = self.t_mean * 60
        ls = LombScargle(time_seconds, self.flux_bgsub)
        freq, amp = ls.autopower(nyquist_factor=1)

        detrad = self.detrad
        ls_detrad = LombScargle(time_seconds, detrad)
        freq_detrad, amp_detrad = ls_detrad.autopower(nyquist_factor=1)

        pgram_tup = WDranker_2.find_cPGRAM(ls, amp_detrad, 
                                           exposure=self.exposure)
        c_periodogram = pgram_tup.c
        if c_periodogram > 0:
            return True
        else:
            return False

def main(csvname):
    allobject = AllLC(csvname)
    allobject.itervisits()


def detectabilitywrapper(noreplace):
    pool = mp.Pool(processes=mp.cpu_count()+2)
    jobs = []
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.csv'):
            if noreplace and os.path.isfile(
                    "pickles/"+filename[:-4]+".pickle"):
                print("Output for "+filename[:-4]+" already exists, skipping")
                print('-'*100)
                continue
            else:
                job = pool.apply_async(main, args=(filename))
                jobs.append(job)

    for job in jobs:
        job.get()


if __name__ == '__main__':
    """
    allobject = AllLC('SDSS-J222816.29+134714.4-NUV.csv')
    visit3 = visit(allobject, 3)
    visit3.df_params()
    synth = SyntheticVisit(visit3)
    #synth.inject(12, 5, 200)
    #value = synth.assessrecovery()
    synth.detectabilitylimit()
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--csvname",
                        help="CSV name corresponding to source", 
                        type=str, default=None)
    parser.add_argument("--wrapper", 
                        help="Multiprocessing of entire directory", 
                        default=False, action='store_true')
    parser.add_argument("--noreplace",
                        help="Continue rather than overwriting",
                        default=False, action='store_true')
    args = parser.parse_args()
    
    if args.csvname is not None:
        main(args.csvname)
    else:
        assert(args.wrapper==True)
        detectabilitywrapper(noreplace=args.noreplace)
