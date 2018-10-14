#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
from astropy.stats import LombScargle
from astropy.stats import median_absolute_deviation
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
random.seed()
from scipy.optimize import curve_fit
import WDranker_2
import WDutils
from WD_Recovery import calculate_jd, selectOptical

# Dom Rowan REU 2018

# Define a class for a galex visit df
class Visit:
    def __init__(self, df, filename, mag=None):
        # Argument assertions
        if type(df) != pd.core.frame.DataFrame:
            raise TypeError("df must be a pandas DataFrame")
        for c in ['t_mean', 'flux_bgsub', 'flux_bgsub_err', 'mag_bgsub']:
            if c not in df.columns:
                raise TypeError("df must be formatted from gPhoton output")
                break
        if (type(filename) != str) or (filename[-3:] != 'csv'):
            raise TypeError("filename must be string ending in .csv")
        self.df = df
        self.timereset = False
        self.df = self.df.reset_index(drop=True)
        self.filename = filename
        self.mag = mag
        # Determine band
        if 'FUV' in self.filename:
            self.band = 'FUV'
            self.fuv_filename = self.filename
            self.nuv_filename = self.fuv_filename.replace('FUV', 'NUV')
        else:
            if 'NUV' not in self.filename:
                raise ValueError('Filename must include band info')
            else:
                self.band = 'NUV'
                self.nuv_filename = self.filename
                self.fuv_filename = self.nuv_filename.replace('FUV', 'NUV')
            
        self.cEXP()
        self.original_median = self.flux_median()

        self.fuv_path = 'FUV/'
        self.nuv_path = ""

    # Calculate exposure using c_EXP metric
    def cEXP(self):
        tup = WDranker_2.find_cEXP(self.df)
        self.c_exposure = tup.c_exposure
        self.exposure = tup.exposure

    # Calculate the median flux_bgsub
    def flux_median(self):
        df_reduced = WDutils.df_fullreduce(self.df)
        allflux = df_reduced['flux_bgsub']
        median = np.nanmedian(allflux)
        return median

    # Simple check if we have enough usable data
    def good_df(self):
        # Create a second data frame with fully reduced data
        # Use this as a way to check flag points
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

    # Reset time to minutes from start
    def reset_time(self):
        for i in range(len(self.df['t_mean'])):
            jd = calculate_jd(self.df['t_mean'][i])

            if i == 0:
                tmin = jd

            if i == len(self.df['t_mean']) - 1:
                tmax = jd

            newtime = (jd - tmin) * 1440
            self.df.loc[i, 't_mean'] = newtime
        self.timereset = True
        self.tmin = tmin
        self.tmax = tmax

    def set_FUVdir(self, path):
        if type(path) != str:
            raise TypeError("FUV path must be string")
        if path[-1] != '/':
            self.fuv_path = f"{path}/"
        else:
            self.fuv_path = path

    def set_NUVdir(self, path):
        if type(path) != str:
            raise TypeError("FUV path must be string")
        if path[-1] != '/':
            self.nuv_path = f"{path}/"
        else:
            self.nuv_path = path

    def FUVexists(self):
        if os.path.isfile(f"{self.fuv_path}{self.fuv_filename}"):
            return True
        else:
            return False

    def NUVexists(self, path=None):
        if os.path.isfile(f"{self.nuv_path}{self.fuv_filename}"):
            return True
        else:
            return False

    def FUVmatch(self):
        if self.FUVexists():
            alldata_fuv = pd.read_csv(f"{self.fuv_path}{self.fuv_filename}")
            alldata_fuv = WDutils.df_fullreduce(alldata_fuv)
            alldata_fuv = WDutils.tmean_correction(alldata_fuv)
            fuv_relativetup = WDutils.relativescales_1(alldata_fuv)
            alldata_fuv_t_mean = fuv_relativetup.t_mean
            alldata_fuv_flux = fuv_relativetup.flux
            alldata_fuv_flux_err = fuv_relativetup.err

            if self.timereset == False:
                self.reset_time()
            for i in range(len(alldata_fuv['t_mean'])):
                jd = calculate_jd(alldata_fuv['t_mean'][i])
                alldata_fuv.loc[i, 't_mean'] = jd
            fuv_relativetup = WDutils.relativescales_1(alldata_fuv)
            alldata_fuv_t_mean = fuv_relativetup.t_mean
            alldata_fuv_flux = fuv_relativetup.flux
            alldata_fuv_flux_err = fuv_relativetup.err

            idx_fuv = np.where((alldata_fuv_t_mean >= self.tmin)
                               & (alldata_fuv_t_mean <= self.tmax))[0]
            t_mean_fuv = np.array(alldata_fuv_t_mean[idx_fuv])
            flux_fuv = np.array(alldata_fuv_flux[idx_fuv])
            flux_err_fuv = np.array(alldata_fuv_flux_err[idx_fuv])

            t_mean_fuv = [(jd - self.tmin) * 1440 for jd in t_mean_fuv]

            OutputTup = collections.namedtuple('OutputTup', ['t_mean',
                                                             'flux',
                                                             'err'])
            tup = OutputTup(t_mean_fuv, flux_fuv, flux_err_fuv)

            return tup
        else:
            return None

    # Inject an optical lc and scale by multiplicative factor
    def inject(self, opticalLC, mf, plot=False, center=None):
        if self.timereset == False:
            self.reset_time()

        if self.FUVexists():
            exists = True
            fuv_tup = self.FUVmatch()
            self.t_mean_fuv = fuv_tup.t_mean
            flux_fuv = fuv_tup.flux
            flux_err_fuv = fuv_tup.err
        else:
            exists = False

        t_mean = self.df['t_mean']
        # Shift to relative flux scales
        relativetup = WDutils.relativescales_1(self.df)
        flux = relativetup.flux
        flux_err = relativetup.err

        if plot:
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6))
            ax0.errorbar(t_mean, flux, yerr=flux_err, color='red')
            if exists:
                ax0.errorbar(self.t_mean_fuv, flux_fuv,
                             yerr=flux_err_fuv, color='blue')
            ax0.axhline(np.median(flux), color='black', ls='--')
            ax0.set_title("Original LC")
            ax0 = WDutils.plotparams(ax0)

        # Get optical data
        df_optical, ax_optical = selectOptical(
            opticalLC,
            exposure=self.exposure / 60,
            plot=plot, center=center)

        # linear interpolation to match times
        optical_flux = np.interp(t_mean, df_optical['time'],
                                 df_optical['flux'])

        if exists:
            optical_flux_fuv = np.interp(self.t_mean_fuv, df_optical['time'],
                                         df_optical['flux'])

        # Scale by multiplicative factor
        optical_flux = [(o - 1) * mf + 1 for o in optical_flux]
        flux_injected = [flux[i] * optical_flux[i]
                         for i in range(len(flux))]

        # Put back into df
        self.df.loc[:, 'flux_bgsub'] = flux_injected
        self.df.loc[:, 'flux_bgsub_err'] = flux_err

        # Do the same for the FUV
        if exists:
            optical_flux_fuv = [(o - 1) * mf + 1 for o in optical_flux_fuv]
            flux_injected_fuv = [flux_fuv[i] * optical_flux_fuv[i]
                                 for i in range(len(flux_fuv))]
            self.flux_injected_fuv = flux_injected_fuv
            self.flux_err_fuv = flux_err_fuv

        # Remove colored points (flag, expt, sigma clip)
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
                ax1.errorbar(self.t_mean_fuv, self.flux_injected_fuv,
                             yerr=self.flux_err_fuv, color='blue')
            ax1.axhline(np.median(self.flux_injected), color='black', ls='--')
            ax1.set_title("Injected LC")
            ax1 = WDutils.plotparams(ax1)

            # plt.show()
            return ax1

    # Use the periodogram metric to test if injected signal is recoverable
    def assessrecovery(self):
        exists = self.FUVexists()

        # Exposure metric already computed in init (self.c_exposure)

        # Periodogram Metric
        time_seconds = self.df['t_mean'] * 60
        ls = LombScargle(time_seconds, self.flux_injected)
        freq, amp = ls.autopower(nyquist_factor=1)

        detrad = self.df['detrad']
        ls_detrad = LombScargle(time_seconds, detrad)
        freq_detrad, amp_detrad = ls_detrad.autopower(nyquist_factor=1)
        pgram_tup = WDranker_2.find_cPGRAM(ls, amp_detrad,
                                           exposure=self.exposure)
        # Return 0,1 rseult of recovery
        c_periodogram = pgram_tup.c
        ditherperiod_exists = pgram_tup.ditherperiod_exists

        # Welch Stetson Metric
        if exists:
            c_ws = WDranker_2.find_cWS(self.t_mean, self.t_mean_fuv,
                                       self.flux_injected,
                                       self.flux_injected_fuv,
                                       self.flux_err, self.flux_err_fuv,
                                       ditherperiod_exists, self.FUVexists())
        else:
            c_ws = WDranker_2.find_cWS(self.t_mean, None,
                                       self.flux_injected, None,
                                       self.flux_err, None,
                                       ditherperiod_exists, self.FUVexists())

        # RMS Metric --- have to 'unscale' the magnitudes
        converted_flux = [f * self.original_median
                          for f in self.flux_injected]
        injectedmags = [WDutils.flux_to_mag('NUV', f)
                        for f in converted_flux]
        sigma_mag = median_absolute_deviation(injectedmags)
        c_magfit = WDranker_2.find_cRMS(self.mag, sigma_mag, 'NUV')

        # Weights:
        w_pgram = 1
        w_expt = .2
        w_WS = .3
        w_magfit = .25

        C = ((w_pgram * c_periodogram)
             + (w_expt * self.c_exposure)
             + (w_magfit * c_magfit)
             + (w_WS * c_ws))

        # cutoff = FindCutoff(95)
        cutoff = .638  # Don't waste time loading in alldata
        # print("Rank --- ", C, "Cutoff --- ", cutoff)

        if C > cutoff:
            return 1
        else:
            return 0

    # See if we have an exisiting period
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
        strongest_period_tup = pgram_tup.strongest_period_tup
        if strongest_period_tup[0] != -1:
            self.period = strongest_period_tup[0]
        else:
            self.period = float('NaN')
        c_periodogram = pgram_tup.c
        if c_periodogram > 0:
            return True
        else:
            return False

    def sinefit(self):
        if not self.timereset:
            self.reset_time()

        coloredtup = WDutils.ColoredPoints(self.df)
        redpoints = coloredtup.redpoints
        bluepoints = coloredtup.bluepoints
        droppoints = np.unique(np.concatenate([redpoints, bluepoints]))
        df_reduced = self.df.drop(index=droppoints)
        df_reduced = df_reduced.reset_index(drop=True)
        df_reduced = WDutils.df_firstlast(df_reduced)
        relativetup = WDutils.relativescales(df_reduced)
        t_mean = relativetup.t_mean
        flux_bgsub = relativetup.flux
        ls = LombScargle(t_mean, flux_bgsub)
        freq, amp = ls.autopower(nyquist_factor=1)
        freq_max = freq[np.where(np.array(amp) == max(amp))[0][0]]
        period_guess = (1 / freq_max)

        time_seconds = t_mean * 60

        def fitfunc(t, A, b, phi):
            return A * np.sin(b * t + phi)

        p0 = [np.max(flux_bgsub), 2 * np.pi * period_guess, 0]
        popt, pcov = curve_fit(fitfunc, time_seconds,
                               flux_bgsub, p0=p0, absolute_sigma=False)

        return (popt[1] / (2 * np.pi)) * 60, pcov

    # Simple Representation
    def __repr__(self):
        title = f"GALEX visit from {self.filename}"
        band = f"Band: {self.band}"
        dfinfo = (f"{self.df.shape[1]} Columns with {self.df.shape[0]} rows")
        return f"{title}\n{band}\n{dfinfo}"

