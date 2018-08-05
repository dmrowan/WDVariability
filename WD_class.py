#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import LombScargle
import heapq
import matplotlib.image as mpimg
import matplotlib.gridspec as gs
import subprocess
from gPhoton import gphoton_utils
import math
from WDutils import readASASSN, badflag_bool, catalog_match
#Dom Rowan REU 2018

desc="""
WD_class: Upgrade on WD ranker, generate class for WD containing rank info.
Rank value for a source dependent on four metrics
"""

class WD:
    def __init__(csvname):
        ###Path assertions###
        catalogpath = ("/home/dmrowan/WhiteDwarfs/"
                      +"Catalogs/MainCatalog_reduced_simbad_asassn.csv")
        assert(os.path.isfile(csvname))
        assert(os.path.isfile(catalogpath))
        assert(os.path.isdir('PDFs'))
        assert(os.path.isdir('Output'))

        csvpath = csvname
        csvpath = csvname
        for i in range(len(csvpath)):
            character = csvpath[i]
            if character == 'c':
                endidx=i-5
                break
        self.source = csvpath[0:endidx]
        #Get band
        if csvpath[-7] == 'N':
            self.band = 'NUV'
            self.band_other = 'FUV'
        elif csvpath[-7] == 'F':
            self.band = 'FUV'
            self.band_other = 'NUV'
        else:
            print("Not source csv, skipping")
            return

        self.alldata = pd.read_csv(csvpath)

        #Catalog Query
        bigcatalog = pd.read_csv(catalogpath)
        bigcatalog_idx = catalog_match(source, bigcatalog)
        if len(bigcatalog_idx) == 0:
            print(source, "Not in catalog")
        else:
            bigcatalog_idx = bigcatalog_idx[0]
            self.spectype = bigcatalog['spectype'][bigcatalog_idx]
            self.variability = bigcatalog['variability'][bigcatalog_idx]
            self.binarity = bigcatalog['binarity'][bigcatalog_idx]
            self.hasdisk = bigcatalog['hasdisk'][bigcatalog_idx]
            self.simbad_name = bigcatalog['SimbadName'][bigcatalog_idx]
            self.simbad_types = bigcatalog['SimbadTypes'][bigcatalog_idx]
            self.gmag = bigcatalog['gaia_g_mean_mag'][bigcatalog_idx]

    
    def reduce(self):
        ###Alldata table corrections###
        #Drop rows with > 10e10 in cps, cps_err, cps < .5
        idx_high_cps = np.where( 
                (self.alldata['cps_bgsub'] > 10e10) 
                | (self.alldata['cps_bgsub_err'] > 10e10) 
                | (self.alldata['counts'] < 1) 
                | (self.alldata['counts'] > 100000) 
                | (self.alldata['flux_bgsub'] < 0) 
                | (self.alldata['cps_bgsub'] < -10000) )[0]
        if len(idx_high_cps) != 0:
            self.alldata = self.alldata.drop(index = idx_high_cps)
            self.alldata = self.alldata.reset_index(drop=True)

        #Fix rows with incorrecct t_means by averaging t0 and t1
        idx_tmean_fix = np.where( 
                (self.alldata['t_mean'] < 1) 
                | (self.alldata['t_mean'] > self.alldata['t1']) 
                | (np.isnan(self.alldata['t_mean'])) )[0]

        for idx in idx_tmean_fix:
            t0 = self.alldata['t0'][idx]
            t1 = self.alldata['t1'][idx]
            mean = (t1 + t0) / 2.0
            self.alldata['t_mean'][idx] = mean

    def other_reduce(self):
        ###See if we have any data in the other band###
        csvpath_other = list(csvpath)
        csvpath_other[-7] = band_other[0]
        csvpath_other = "".join(csvpath_other)
        #Look for file in GALEXphot/LCs
        csvpath_other = ('/home/dmrowan/WhiteDwarfs/GALEXphot/LCs/'
                         +csvpath_other)
        if os.path.isfile(csvpath_other):
            other_band_exists = True
            alldata_other = pd.read_csv(csvpath_other)
        else:
            other_band_exists = False

        if other_band_exists:
            #print("Generating additional LC data for " 
                    #+ band_other + " band")
            alldata_other = pd.read_csv(csvpath_other)
            #Drop bad rows, flagged rows
            idx_high_cps_other = np.where( 
                    (alldata_other['cps_bgsub'] > 10e10)
                    | (alldata_other['cps_bgsub_err'] > 10e10)
                    | (alldata_other['counts'] < 1)
                    | (alldata_other['counts'] > 100000)
                    | (alldata_other['flux_bgsub'] < 0)
                    | (alldata_other['cps_bgsub'] < -10000) )[0]
            idx_other_flagged_bool = [ badflag_bool(x)
                                       for x in alldata_other['flags'] ]
            idx_other_flagged = np.where(
                    np.array(idx_other_flagged_bool) == True)[0]
            idx_other_expt = np.where(alldata_other['exptime'] < 10)[0]

            idx_other_todrop = np.unique(np.concatenate([idx_high_cps_other,
                                                         idx_other_flagged,
                                                         idx_other_expt]))
            alldata_other = alldata_other.drop(index=idx_other_todrop)
            alldata_other = alldata_other.reset_index(drop=True)

            stdev_other = np.std(alldata_other['flux_bgsub'])
            if len(alldata_other['flux_bgsub']) != 0:
                if not alldata_other['flux_bgsub'].isnull().all():
                    idx_fivesigma_other = np.where(
                            abs((alldata_other['flux_bgsub']
                                - np.nanmean(alldata_other['flux_bgsub'])))
                            > 5*stdev_other )[0]
                    alldata_other = alldata_other.drop(
                            index=idx_fivesigma_other)
                    alldata_other = alldata_other.reset_index(drop=True)

            #Fix rows with weird t_mean time
            idx_tmean_fix_other = np.where( 
                    (alldata_other['t_mean'] < 1) 
                    | (alldata_other['t_mean'] > alldata_other['t1']) 
                    | (np.isnan(alldata_other['t_mean'])) )[0]
            for idx in idx_tmean_fix_other:
                t0 = alldata_other['t0'][idx]
                t1 = alldata_other['t1'][idx]
                mean = (t1 + t0) / 2.0
                alldata_other['t_mean'][idx] = mean

            #Make correction for relative scales
            self.alldata_tmean_other = alldata_other['t_mean']
            alldata_flux_bgsub_other = alldata_other['flux_bgsub']
            alldata_medianflux_other = np.median(alldata_flux_bgsub_other)
            self.alldata_flux_bgsub_other = (
                    (alldata_flux_bgsub_other /
                        alldata_medianflux_other ) - 1.0) * 100
            self.alldata_flux_bgsub_err_other = (
                    alldata_other['flux_bgsub_err']
                    / alldata_medianflux_other) * 100

    #Magnitude and RMS mag info
    def magnitude(self):
        #Path Assertions
        sigmamag_path_NUV = "Catalog/SigmaMag_NUV.csv"
        sigmamag_percentile_path_NUV = "Catalog/magpercentiles_NUV.csv"
        sigmamag_path_FUV = "Catalog/SigmaMag_FUV.csv"
        sigmamag_percentile_path_FUV = "Catalog/magpercentiles_FUV.csv"
        assert(os.path.isfile(sigmamag_path_NUV))
        assert(os.path.isfile(sigmamag_percentile_path_NUV))
        assert(os.path.isfile(sigmamag_path_FUV))
        assert(os.path.isfile(sigmamag_percentile_path_FUV))
        ###Apparent Magnitude### 
        self.m_ab = np.nanmedian(self.alldata['mag_bgsub'])
        sigma_mag_all = np.nanstd( (self.alldata['mag_bgsub_err_1']
                + alldata['mag_bgsub_err_2'])/2.0 )
        #Calculate c_mag based on ranges:
        if self.m_ab > 13 and self.m_ab < 25:
            self.c_mag = self.m_ab**(-1) * 10
        else:
            self.c_mag = 0

        self.magdic = {"mag":[m_ab], "sigma":[sigma_mag_all], "weight":[1]}

        #Read in mag percentile information
        if self.band == 'NUV':
            percentile_df = pd.read_csv(sigmamag_percentile_path_NUV)
        else:
            assert(self.band == 'FUV')
            percentile_df = pd.read_csv(sigmamag_percentile_path_FUV)

        magbins = percentile_df['magbin']
        magbins = np.array(magbins)
        percentile50 = percentile_df['median']
        lowerbound = percentile_df['lower']
        upperbound = percentile_df['upper']
        if m_ab < 20.75:
            sigmamag_idx = np.where(
                    abs(m_ab-magbins) == min(abs(m_ab-magbins)))[0]
            sigmafit_val = float(percentile50[sigmamag_idx])
            sigmafit_val_upper = float(upperbound[sigmamag_idx])
            if ((sigma_mag_all > sigmafit_val)
                    and (sigma_mag_all < sigmafit_val_upper)):
                self.c_magfit = sigma_mag_all / sigmafit_val
            elif sigma_mag_all >= sigmafit_val_upper:
                self.c_magfit = sigmafit_val_upper / sigmafit_val
            else:
                self.c_magfit = 0
        else:
            self.c_magfit = 0


    def rank(self):
        fap=.05
        prange=.0005
        w_pgram=1
        w_expt=.2
        w_WS=.30
        w_magfit=.25
        c








