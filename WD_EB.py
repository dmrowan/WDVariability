#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import matplotlib.gridspec as gs
from gPhoton import gphoton_utils
from WDranker_2 import badflag_bool, catalog_match
from matplotlib.patheffects import withStroke
import _pickle as pickle
from scipy import optimize
from scipy.stats import norm, chisquare
from progressbar import ProgressBar
import _pickle as pickle

desc="""
WD_EB: Define a class for EBs
"""
#Fit functions
def piecewise_linear_half(x, x0, x1, m1, m2, m3, y0):
    return np.piecewise(x, [x<x0, ((x>x0)&(x<x1))],
                        [lambda x: m1*(x-x0) + y0,
                         lambda x: m2*(x-x0) + y0,
                         lambda x: m3*(x-x1) + m2*(x1-x0) +y0])

def piecewise_linear_full(x, x0,x1,x2, x3, m1,m2, m3, y0):
    return np.piecewise(x, [x<x0, ((x>x0) & (x<x1)),
                            ((x>x1) & (x<x2)), ((x>x2) & (x<x3)) ],
                        [lambda x: m1*x +y0-m1*x0,
                        lambda x: m2*x + y0-m2*x0,
                        lambda x: m3*(x-x1) + m2*(x1-x0) + y0,
                        lambda x: (-m2)*(x-x2) + m3*(x2-x1) + m2*(x1-x0)+y0,
                        lambda x: m1*(x-x3) + (-m2)*(x3-x2) +
                                  m3*(x2-x1) + m2*(x1-x0)+y0])

def piecewise_linear_V(x, x0, x1, x2, m1, m2, y0):
    return np.piecewise(x, [x<x0, ((x>x0) & (x<x1)), ((x>x1) & (x<x2))],
                        [lambda x: m1*(x-x0) + y0, 
                         lambda x: m2*(x-x0) + y0, 
                         lambda x: (-m2)*(x-x1) + m2*(x1-x0) + y0,
                         lambda x: m1*(x-x2) + (-m2)*(x2-x1) 
                                 + m2*(x1-x0) +y0])


class WDEB:
    
    def __init__(self, NUVcsv):
        #Read in CSV and get source name
        #Generate DataFrame for both bands
        self.NUVcsv = NUVcsv
        assert(os.path.isfile(self.NUVcsv))
        self.NUVall = pd.read_csv(self.NUVcsv)
        self.source = self.NUVcsv[:-8]

        FUVcsv = self.NUVcsv.replace('NUV', 'FUV')
        if os.path.isfile(FUVcsv):
            self.FUVcsv = FUVcsv
            self.FUVall = pd.read_csv(self.FUVcsv)
        else:
            self.FUVcsv = None
            self.FUVall = None

        #Best group information
        group_path = ("/home/dmrowan/WhiteDwarfs/" +
                      "InterestingSources/eclipse_group.csv")
        assert(os.path.isfile(group_path))
        df_groups = pd.read_csv(group_path)

        df_groups = df_groups.where(pd.notnull(df_groups), None)
        idx_group = np.where(df_groups['MainID']==self.source)[0][0]
        self.best_df = df_groups['df_number'][idx_group]
        self.g1 = df_groups['g1'][idx_group]
        self.g2 = df_groups['g2'][idx_group]
        self.g3 = df_groups['g3'][idx_group]
        self.g4 = df_groups['g4'][idx_group]

        
        #Catalog match
        bigcatalog = pd.read_csv("/home/dmrowan/WhiteDwarfs/Catalogs/"
                                 +"MainCatalog_reduced_simbad_asassn.csv")
        bc_idx = catalog_match(self.source, bigcatalog)
        if len(bc_idx) == 0:
            print(self.source, "Not in catalog")
        else:
            self.ra = bigcatalog['ra'][bc_idx]
            self.dec = bigcatalog['dec'][bc_idx]
            self.sdsstype = bigcatalog['spectype'][bc_idx]
            self.mwddbinary = bigcatalog['binarity'][bc_idx]
            self.simbadtypes = str(bigcatalog['SimbadTypes'][bc_idx])
            if 'EB' in self.simbadtypes:
                self.knownEB = True
            else:
                self.knownEB = False

        #IS.csv match (from WDAssign)
        ISpath='/home/dmrowan/WhiteDwarfs/InterestingSources/IS.csv'
        assert(os.path.isfile(ISpath))
        df_IS = pd.read_csv(ISpath)
        idx_IS = np.where(df_IS["MainID"] == self.source)[0][0]
        self.labelnum = df_IS['labelnum'][idx_IS]
    #Normal reduction pipeline to get rid of gphoton issues
    def reduce(self):
        #Remove bad lines from gPhoton
        idx_cut = np.where( (abs(self.NUVall['cps_bgsub']) > 10e10) 
                                 | (self.NUVall['cps_bgsub_err'] > 10e10) 
                                 | (self.NUVall['counts'] < 1) 
                                 | (self.NUVall['counts'] > 100000) )[0]

        if len(idx_cut) != 0:
            self.NUVall = self.NUVall.drop(index = idx_cut)
            self.NUVall = self.NUVall.reset_index(drop=True)

        #Fix t_mean column when necessary
        idx_tmean_fix = np.where( (self.NUVall['t_mean'] < 1) 
                                | (self.NUVall['t_mean'] > self.NUVall['t1']) 
                                | (np.isnan(self.NUVall['t_mean'])) )[0]
        for idx in idx_tmean_fix:
            t0 = self.NUVall['t0'][idx]
            t1 = self.NUVall['t1'][idx]
            mean = (t1 + t0) / 2.0
            self.NUVall['t_mean'][idx] = mean

        if self.FUVall is not None:
            #Remove bad lines from gPhoton
            idx_FUV_cut = np.where( (abs(self.FUVall['cps_bgsub']) > 10e10) 
                                  | (self.FUVall['cps_bgsub_err'] > 10e10) 
                                  | (self.FUVall['counts'] < 1) 
                                  | (self.FUVall['counts'] > 100000) )[0]

            idx_FUV_flag_bool = np.array([ badflag_bool(x) 
                                           for x in self.FUVall['flags'] ])

            idx_FUV_flagged = np.where(idx_FUV_flag_bool == True)[0]

            idx_FUV_expt = np.where(self.FUVall['exptime'] < 10)[0]
            
            idx_FUV_drop = np.unique(np.concatenate([idx_FUV_cut, 
                                                     idx_FUV_flagged, 
                                                     idx_FUV_expt]))

            self.FUVall = self.FUVall.drop(index=idx_FUV_drop)
            self.FUVall = self.FUVall.reset_index(drop=True)

            #Fix t_mean column when necessary
            idx_tmean_fix_FUV = np.where( (self.FUVall['t_mean'] < 1)
                        | (self.FUVall['t_mean'] > self.FUVall['t1'])
                        | (np.isnan(self.FUVall['t_mean'])) )[0]
            for idx in idx_tmean_fix_FUV:
                t0 = self.FUVall['t0'][idx]
                t1 = self.NUVall['t1'][idx]
                mean = (t1 + t0) / 2.0
                self.FUVall['t_mean'][idx] = mean

        #Split up NUV df, select from best number
        breaks = []
        for i in range(len(self.NUVall['t0'])):
            if i != 0:
                if (self.NUVall['t0'][i] - self.NUVall['t0'][i-1]) >= 200:
                    breaks.append(i)

        data = np.split(self.NUVall, breaks)
        df = data[self.best_df-1]

        lasttime = list(df['t1'])[-1]
        firsttime = list(df['t0'])[0]

        ###Dataframe corrections###
        #Reset first time in t_mean to be 0
        firsttime_mean = df['t_mean'][df.index[0]]
        df['t_mean'] = df['t_mean'] - firsttime_mean

        #Red points and bluepoints
        stdev = np.std(df['cps_bgsub'])
        bluepoints = np.where( 
                (df['cps_bgsub'] - np.nanmean(df['cps_bgsub'])) > 5*stdev )[0]
        flag_bool_vals = [ badflag_bool(x) for x in df['flags'] ]
        redpoints1 = np.where(np.array(flag_bool_vals) == True)[0]
        redpoints2 = np.where(df['exptime'] < 5)[0]
        redpoints = np.unique(np.concatenate([redpoints1, redpoints2]))
        self.redpoints = redpoints + df.index[0]
        self.bluepoints = bluepoints + df.index[0]

        droppoints = np.unique(np.concatenate([self.redpoints, 
                                               self.bluepoints]))
        df_reduced = df.drop(index=droppoints)
        df_reduced = df_reduced.reset_index(drop=True)

        #Remove points where cps_bgsub is nan
        idx_cps_nan = np.where( np.isnan(df_reduced['cps_bgsub']) )[0]
        if len(idx_cps_nan) != 0:
            df_reduced = df_reduced.drop(index=idx_cps_nan)
            df_reduced = df_reduced.reset_index(drop=True)

        #If first point is not within 3 sigma, remove
        if (df_reduced['cps_bgsub'][df_reduced.index[0]] 
                                    - np.nanmean(df['cps_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[0])

        #If last point is not within 3 sigma, remove
        if (df_reduced['cps_bgsub'][df_reduced.index[-1]] 
                - np.nanmean(df['cps_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[-1])

        #Load t_mean, cps, cps_err into object
        self.cps_bgsub = df_reduced['cps_bgsub']
        cps_bgsub_median = np.median(self.cps_bgsub)
        self.cps_bgsub = self.cps_bgsub / cps_bgsub_median
        self.cps_bgsub_err = df_reduced['cps_bgsub_err'] / cps_bgsub_median
        self.t_mean = df_reduced['t_mean']

        self.jd_t_mean = [ gphoton_utils.calculate_jd(t+firsttime_mean)
                           for t in self.t_mean ]
        tmin = min(self.jd_t_mean)
        self.jd_t_mean = [ (t - tmin)*1440 for t in self.jd_t_mean ]
        self.obsstart = tmin

        #If we have data in the other band, match it up
        if self.FUVall is not None:
            self.t_mean_FUV = self.FUVall['t_mean']
            self.cps_bgsub_FUV = self.FUVall['cps_bgsub']
            cps_bgsub_median_FUV = np.median(self.cps_bgsub_FUV)
            self.cps_bgsub_FUV = self.cps_bgsub_FUV / cps_bgsub_median_FUV
            self.cps_bgsub_err_FUV = (
                    self.FUVall['cps_bgsub_err'] / cps_bgsub_median_FUV)
            
            idx_expgroup_FUV = np.where(
                    (self.t_mean_FUV > firsttime) &
                    (self.t_mean_FUV < lasttime) )[0]

            self.t_mean_FUV = (self.t_mean_FUV[idx_expgroup_FUV] 
                               - firsttime_mean)

            self.cps_bgsub_FUV = self.cps_bgsub_FUV[idx_expgroup_FUV]
            self.cps_bgsub_err_FUV = self.cps_bgsub_err_FUV[idx_expgroup_FUV]

            self.jd_t_mean_FUV = [
                    (gphoton_utils.calculate_jd(t+firsttime_mean) - tmin)*1440
                    for t in self.t_mean_FUV ]

        #Get relative scales for redpoints and bluepoints and plot
        if len(self.redpoints) != 0:
            self.cps_bgsub_red = (df['cps_bgsub'][self.redpoints] 
                                  / cps_bgsub_median)
            self.cps_bgsub_err_red = (df['cps_bgsub_err'][self.redpoints] 
                                      / cps_bgsub_median)
            self.t_mean_red = df['t_mean'][self.redpoints]
            self.jd_t_mean_red = [ 
                    (gphoton_utils.calculate_jd(t+firsttime_mean) - tmin)*1440
                    for t in self.t_mean_red ]

            
        if len(self.bluepoints) != 0:
            self.cps_bgsub_blue = (df['cps_bgsub'][self.bluepoints] 
                                   / cps_bgsub_median)
            self.cps_bgsub_err_blue = (df['cps_bgsub_err'][self.bluepoints]
                                       / cps_bgsub_median)
            self.t_mean_blue = df['t_mean'][self.bluepoints]
            self.jd_t_mean_blue = [ 
                    (gphoton_utils.calculate_jd(t+firsttime_mean) - tmin)*1440
                    for t in self.t_mean_blue ]

            
    #Generate matplotlib LC
    def LCplot(self, save=True):   
        #Generate plot information
        self.reduce()
        #Generate LC plot
        fig, ax = plt.subplots(1,1)
        ax.errorbar(self.jd_t_mean, self.cps_bgsub, yerr=self.cps_bgsub_err,
                    color='red', marker='.', ls='', zorder=4, label='NUV')
        ax.axhline(y=0, alpha=.3, ls='dotted', color='red')

        #Plot redpoints and bluepoints
        if len(self.redpoints) != 0:
            ax.errorbar(self.jd_t_mean_red, self.cps_bgsub_red,
                        yerr=self.cps_bgsub_err_red, color='#808080',
                        marker='.', ls='', zorder=2, alpha=.5,
                        label='Flagged')

        if len(self.bluepoints) != 0:
            ax.errorbar(self.jd_t_mean_blue, self.cps_bgsub_blue,
                        yerr=self.cps_bgsub_err_blue, color='green', 
                        marker='.', ls='', zorder=3, alpha=.5, 
                        label='SigmaClip')

        #Plot other band if exists
        if self.FUVall is not None:
            if len(self.t_mean_FUV) > 0:
                ax.errorbar(self.jd_t_mean_FUV, self.cps_bgsub_FUV, 
                            yerr=self.cps_bgsub_err_FUV,
                            color='blue', marker='.', ls='',
                            zorder=1, label='FUV', alpha=.5)


        #Plot params
        myeffectw = withStroke(foreground='black', linewidth=2)
        txtkwargsw = dict(path_effects=[myeffectw])
        afont = {'fontname':'Keraleeyam'}
        ax.set_xlabel('Time Elapsed (min)', fontsize=20)
        ax.set_ylabel('Relative CPS', fontsize=20)
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', labelsize=12)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params('both', length=8, width=1.8, which='major')
        ax.tick_params('both', length=4, width=1, which='minor')
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

        ax.annotate(str(self.labelnum), xy=(.95, .95), 
                    xycoords='axes fraction', color='xkcd:azure', 
                    fontsize=30, ha='center', va='center',
                    **afont, **txtkwargsw)

        self.LCfig = fig
        self.LCax = ax
        

        if save:
            self.saveLC()

    #Fit the LC with transit models
    def fit(self, bootstrap=False, cpsvals=None, save=False):
        self.reduce()
        if save:
            self.LCplot()
            fig = self.LCfig
            ax = self.LCax
        
        #fig, ax = self.LCfig, self.LCax
        xarray = np.arange(0, max(self.jd_t_mean), .1)
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        g4 = self.g4
        if bootstrap:
            assert(cpsvals is not None)
        else:
            cpsvals = self.cps_bgsub

        #Generate dictionary to store parameters
        paramdic = {}
        
        #Fit depends on conditions input (ingress, egress, full)
        if not None in [g1,g2,g3,g4]:
            #print("Using full fit function")
            self.fitused = 'full'
            p,e = optimize.curve_fit(
                    piecewise_linear_full, 
                    self.jd_t_mean, cpsvals,
                    sigma=self.cps_bgsub_err,
                    p0=[g1,g2,g3,g4,0,-.5,0,1],
                    bounds=([g1-2,g2-2,g3-2,g4-2,-.5,-2,-.1,0.8],
                            [g1+2,g2+2,g3+2,g4+2,0.5,-.1,0.1,1.2]))

            if save:
                ax.plot(xarray, piecewise_linear_full(xarray, *p),
                        zorder=5, lw=4, color='black')

            expectedvals = piecewise_linear_full(self.jd_t_mean, *p)
            
            paramdic['duration'] = p[3] - p[0]
            paramdic['ingress'] = p[1] - p[0]
            paramdic['egress'] = p[3]- p[2]
            paramdic['reg_slope'] = p[4]
            paramdic['ie_slope'] = p[5]
            paramdic['bot_slope'] = p[6]

        elif ((None not in [g1,g2]) 
                and (all([val == None for val in [g3,g4]]))):
            #print("Using ingress fit function")
            self.fitused = 'half'
            p, e = optimize.curve_fit(
                    piecewise_linear_half,
                    self.jd_t_mean, cpsvals, 
                    sigma=self.cps_bgsub_err,
                    p0=[g1, g2, 0, .5, 0, 1],
                    bounds=([g1-2, g2-2, -.25, -1, -.5, .5],
                            [g1+2, g2+2, 0.25, 1, 0.5, 1.5]))

            if save:
                ax.plot(xarray, piecewise_linear_half(xarray, *p))

            expectedvals = piecewise_linear_half(self.jd_t_mean, *p)

            paramdic['duration'] = None
            paramdic['ingress'] = p[1] - p[0]
            paramdic['egress'] = None
            paramdic['reg_slope'] = p[2]
            paramdic['ie_slope'] = p[3]
            paramdic['bot_slope'] = p[4]

        elif ((all([val == None for val in [g1,g2]]))
                and (None not in [g3,g4])):
            #print("Using egress fit function")
            self.fitused = 'half'
            p, e = optimize.curve_fit(
                    piecewise_linear_half,
                    self.jd_t_mean, cpsvals, 
                    sigma=self.cps_bgsub_err,
                    p0=[g3, g4, 0, .5, 0, 0],
                    bounds=([g3-2, g4-2, -.5, -1, -.5, -.5],
                            [g3+2, g4+2, 0.5, 1, 0.5, 0.5]))

            if save:
                ax.plot(xarray, piecewise_linear_half(xarray, *p))

            expectedvals = piecewise_linear_half(self.jd_t_mean, *p)

            paramdic['duration'] = None
            paramdic['ingress'] = None
            paramdic['egress'] = p[1] - p[0]
            paramdic['reg_slope'] = p[4]
            paramdic['ie_slope'] = p[3]
            paramdic['bot_slope'] = p[2]

        elif (None not in [g1,g2,g4]) and (g3 is None):
            self.fitused='V'
            p, e = optimize.curve_fit(
                    piecewise_linear_V, 
                    self.jd_t_mean, cpsvals,
                    sigma=self.cps_bgsub_err,
                    p0=[g1,g2,g4, 0, -.5, 1],
                    bounds=([g1-2, g2-2, g4-2, -.5, -1, .5],
                            [g1+2, g2+2, g4+2, 0.5, 0.1, 1.5]))

            if save:
                ax.plot(xarray, piecewise_linear_V(xarray, *p))

            expectedvals = piecewise_linear_half(self.jd_t_mean, *p)

            paramdic['duration'] = p[2]-p[0]
            paramdic['ingress'] = p[1] - p[0]
            paramdic['egress'] = p[2]-p[1]
            paramdic['reg_slope'] = p[3]
            paramdic['ie_slope'] = p[4]
            paramdic['bot_slope'] = None
        else:
            print("-----Error in Input Params------")
            
        chi2, p2 = chisquare(cpsvals, expectedvals)
        chi2reduced = chi2 / (len(self.jd_t_mean)-1)
 
        paramdic['chi2reduced'] = chi2reduced

        self.fitparams = p

        if bootstrap:
            return(paramdic)
        else:
            self.bestfit = paramdic
            if save:
                self.LCfig = fig
                self.LCax = ax
                self.saveLC()
            return(self.fitparams)

            print('Reduced Chi2: ',chi2reduced)

    def saveLC(self):
        fig, ax = self.LCfig, self.LCax
        fig.savefig(self.source+"-LCfig.pdf")

    def bootstrap(self, iterations=100):
        self.fit()
        fig2 = plt.figure(figsize=(16,12))
        gs2 = gs.GridSpec(4,2)
        gs2.update(hspace=.3, wspace=.2)
        plt.subplots_adjust(top=.98, right=.98)
        fig2.text(.02, .5, "N in bin", va='center', 
                  rotation='vertical', fontsize=20)
        #Need to exclude some errors that are 0
        adjusted_err = []
        for val in self.cps_bgsub_err:
            if val > .07:
                adjusted_err.append(val)

        #Fit gaussian to errors
        (mu, sigma) = norm.fit(adjusted_err)
        fwhm = 2*np.sqrt(2*np.log(2))*sigma
        lb = mu - (fwhm/2)
        ub = mu + (fwhm/2)
        
        ax2 = plt.subplot(gs2[(0,0)])
        #First plot histogram of err in cps
        n, bins, patches = ax2.hist(adjusted_err, bins=30, density=1,
                                    linewidth=1.2, edgecolor='black',
                                    color='xkcd:azure')
        fit = norm.pdf(bins, mu, sigma)
        ax2.plot(bins, fit, color='red',
                   ls='--', label='Gaussian Fit')
        ax2.axvspan(lb, ub, alpha=.25, color='#808080',
                      label='__nolegend__')
        ax2.axvline(mu, zorder=2, color='black', lw=2,
                      ls='--', label="Median")

        #Plot parameters
        ax2.set_xlabel('Err in counts/second', fontsize=20)
        ax2.minorticks_on()
        ax2.yaxis.set_ticks_position('both')
        ax2.xaxis.set_ticks_position('both')
        ax2.tick_params(direction='in', which='both', labelsize=15)
        ax2.tick_params('both', length=8, width=1.8, which='major')
        ax2.tick_params('both', length=4, width=1, which='minor')
        ax2.legend(loc=2, fontsize=15, edgecolor='black')

        #Initialize dictionaries
        master_dic = {
                'duration':[],
                'ingress':[],
                'egress':[],
                'reg_slope':[],
                'ie_slope':[],
                'bot_slope':[],
                'chi2reduced':[]
            }

        #Output parameters, formatted median, -1sigma, +1sigma
        self.parameters = {
                'duration':[],
                'ingress':[],
                'egress':[],
                'reg_slope':[],
                'ie_slope':[],
                'bot_slope':[],
                'chi2reduced':[]
            }

        #Run iterations
        pbar = ProgressBar()
        for i in pbar(range(0, iterations)):
            cpsvals = []
            for i in range(len(self.cps_bgsub)):
                val = self.cps_bgsub[i]
                err = self.cps_bgsub_err[i]
                val += np.random.normal(val, err)
                cpsvals.append(val)
            try:
                paramdic= self.fit(bootstrap=True, cpsvals=cpsvals)
            except:
                continue
            if paramdic['duration'] is not None:
                master_dic['duration'].append(paramdic['duration'])
            if paramdic['ingress'] is not None:
                master_dic['ingress'].append(paramdic['ingress'])
            if paramdic['egress'] is not None:
                master_dic['egress'].append(paramdic['egress'])
            
            master_dic['reg_slope'].append(paramdic['reg_slope'])
            master_dic['ie_slope'].append(paramdic['ie_slope'])
            master_dic['bot_slope'].append(paramdic['bot_slope'])
            master_dic['chi2reduced'].append(paramdic['chi2reduced'])


        coords = []
        for y in range(0,4):
            coords.append( (y,0) )
            coords.append( (y,1) )
        coords = coords[1:]
        for i in range(len(coords)):
            ax2 = plt.subplot(gs2[coords[i]])
            values_i = master_dic[list(master_dic.keys())[i]]
            if len(values_i) == 0:
                print("No {0} param to plot for {1}".format(
                        list(master_dic.keys())[i], 
                        self.source))

            else:
                n_i, bins_i, patches_i = ax2.hist(
                        values_i, bins=iterations//10,
                        linewidth=1.2, edgecolor='black', 
                        color='xkcd:azure', density=1)
                ax2.set_xlabel(list(master_dic.keys())[i], fontsize=20)

                (mu_i, sigma_i) = norm.fit(values_i)
                fit_i = norm.pdf(bins_i, mu_i, sigma_i)
                ax2.plot(bins_i, fit_i, color='red', ls='--',
                         label='Gaussian fit')

                p_median = np.percentile(values_i, 50)
                p_onesigma = [np.percentile(values_i, p) 
                              for p in [15.9, 84.1]]
                outputparams = [p_median]
                outputparams.extend(p_onesigma)
                self.parameters[list(master_dic.keys())[i]].extend(
                        outputparams)

                ax2.axvline(p_median, color='black', zorder=3,
                            ls='--', label='Median')
                ax2.axvspan(p_onesigma[0], p_onesigma[1], color='#808080',
                            alpha=.25, label='__nolegend__')

            ax2.minorticks_on()
            ax2.yaxis.set_ticks_position('both')
            ax2.xaxis.set_ticks_position('both')
            ax2.tick_params(direction='in', which='both', labelsize=15)
            ax2.tick_params('both', length=8, width=1.8, which='major')
            ax2.tick_params('both', length=4, width=1, which='minor')
            ax2.legend(loc=2, fontsize=15, edgecolor='black')
            ax2.set_ylabel('N in bin', fontsize=20)

        fig2.savefig(self.source+"-bootstrap_hist.pdf")
        self.fig2 = fig2
        self.gs2 = gs2
        return(self.parameters)

    #Generate all useful stuff and save as pickle
    def __call__(self):
        self.bootstrap(iterations=1000)
        return self.parameters


#Generate latex table
def bootstrapwrapper(ii): 
    assert(os.getcwd()=='/home/dmrowan/WhiteDwarfs/InterestingSources')
    dirlist = os.listdir('Eclipse')
    sourcelist = []
    labelnum_list = []
    paramdic_list = []
    eb_list = []
    id_list = []
    for dname in dirlist:
        os.chdir('Eclipse/'+dname)
        eb = WDEB(dname+"-NUV.csv")
        sourcelist.append(eb.source)
        labelnum_list.append(eb.labelnum)
        paramdic_list.append(eb.bootstrap(iterations=ii))
        eb_list.append(eb)
        id_list.append(eb.labelnum)
        os.chdir('../../')

    df_temp = pd.DataFrame({'id':id_list, 'object':eb_list})
    df_temp = df_temp.sort_values(by='id')
    eb_list = list(df_temp['object'])
    with open("EB_Master.pickle", 'wb') as p:
        pickle.dump(eb_list, p)
    
    


#Generate LC tower
def TowerPlot():
    assert(os.getcwd()=='/home/dmrowan/WhiteDwarfs/InterestingSources')
    dirlist = os.listdir('Eclipse')
    eb_list = []
    id_list = []
    for dname in dirlist:
        os.chdir('Eclipse/'+dname)
        eb = WDEB(dname+"-NUV.csv")
        if not eb.knownEB:
            eb_list.append(eb)
            id_list.append(eb.labelnum)
        
        os.chdir('../../')
    
    df_temp = pd.DataFrame({'id':id_list, 'object':eb_list})
    df_temp = df_temp.sort_values(by='id')
    eb_list = list(df_temp['object'])
    figT = plt.figure(figsize=(12,8))
    gsT = gs.GridSpec(4,2)
    gsT.update(hspace=0, wspace=0.15)
    plt.subplots_adjust(top=.98, right=.98)


    myeffectw = withStroke(foreground='black', linewidth=2)
    txtkwargsw = dict(path_effects=[myeffectw])
    afont = {'fontname':'Keraleeyam'}

    figT.text(.02, .5, 'Relative CPS', 
              va='center', rotation='vertical', fontsize=30)
    figT.text(.5, .025, 'Time Elapsed (min)',
             va='center', fontsize=30, ha='center')

    coords = []
    for y in range(0, 4):
        coords.append( (y,0) )
        coords.append( (y,1) )
    idx_plot = 0
    for eb in eb_list:
        if idx_plot == len(eb_list):
            break
        plot_coords = coords[idx_plot]
        params = eb.fit()
        plt.subplot2grid((4,2), plot_coords, colspan=1, rowspan=1)
        axT = plt.subplot(gsT[plot_coords])
        axT.errorbar(eb.jd_t_mean, eb.cps_bgsub, yerr=eb.cps_bgsub_err,
                     color='red', marker='.', ls='', zorder=4, label='NUV')
        axT.axhline(0, alpha=.3, ls='dotted', color='red')

        #Plot redpoints
        if len(eb.redpoints) != 0:
            axT.errorbar(eb.jd_t_mean_red, eb.cps_bgsub_red,
                        yerr=eb.cps_bgsub_err_red, color='#808080',
                        marker='.', ls='', zorder=2, alpha=.5,
                        label='Flagged')

        #Plot other band if exists
        if eb.FUVall is not None:
            if len(eb.t_mean_FUV) > 0:
                axT.errorbar(eb.jd_t_mean_FUV, eb.cps_bgsub_FUV, 
                            yerr=eb.cps_bgsub_err_FUV,
                            color='blue', marker='.', ls='',
                            zorder=1, label='FUV', alpha=.5)

        #Plot fit parameters
        xarray = np.arange(0, max(eb.jd_t_mean), .1)
        if eb.fitused == 'full':
            axT.plot(xarray, piecewise_linear_full(xarray, *eb.fitparams),
                     lw=4, alpha=.5, color='black')

        elif eb.fitused == 'half':
            axT.plot(xarray, piecewise_linear_half(xarray, *eb.fitparams),
                     lw=4, alpha=.5, color='black')
        else:
            assert(eb.fitused == 'V')
            axT.plot(xarray, piecewise_linear_V(xarray, *eb.fitparams),
                     lw=4, alpha=.5, color='black')

        #Plotting parameters
        axT.set_ylim(ymax=max(eb.cps_bgsub)*1.2)
        axT.minorticks_on()
        axT.tick_params(direction='in', which='both', labelsize=12)
        axT.yaxis.set_ticks_position('both')
        axT.xaxis.set_ticks_position('both')
        axT.tick_params('both', length=8, width=1.8, which='major')
        axT.tick_params('both', length=4, width=1, which='minor')
        for axis in ['top', 'bottom', 'left', 'right']:
            axT.spines[axis].set_linewidth(1.5)

        axT.annotate(str(eb.labelnum), xy=(.95, .85), 
                    xycoords='axes fraction', color='xkcd:azure', 
                    fontsize=20, ha='center', va='center', zorder=4
                    **afont, **txtkwargsw)
        idx_plot += 1


    figT.savefig("EclipseTower.pdf")

def genTable():
    assert(os.path.isfile("EB_Master.pickle"))
    with open("EB_Master.pickle", 'rb') as p:
        eb_list = pickle.load(p)
    
    output_dic = {
            'MainID':[],
            'ID': [],
            'Spectral Type':[],
            'Obs Start':[],
            'Duration':[],
            'Ingress':[],
            'Egress':[],
            'Duration_m':[],
            'Duration_bf':[],
            'Duration_lower':[],
            'Duration_upper':[],
            'Ingress_m':[],
            'Ingress_bf':[],
            'Ingress_lower':[],
            'Ingress_upper':[],
            'Egress_m':[],
            'Egress_bf':[],
            'Egress_lower':[],
            'Egress_upper':[],
            }
             
    for eb in eb_list:
        bd = eb.parameters
        output_dic['MainID'].append(eb.source)
        output_dic['ID'].append(eb.labelnum)
        output_dic['Spectral Type'].append(eb.sdsstype)
        output_dic['Obs Start'].append(eb.obsstart)
        output_dic['Duration_m'].append(bd['duration'][0])
        output_dic['Duration_lower'].append(bd['duration'][1])
        output_dic['Duration_upper'].append(bd['duration'][2])
        output_dic['Ingress_m'].append(bd['ingress'][0])
        output_dic['Ingress_lower'].append(bd['ingress'][1])
        output_dic['Ingress_upper'].append(bd['ingress'][2])
        output_dic['Egress_m'].append(bd['egress'][0])
        output_dic['Egress_lower'].append(bd['egress'][1])
        output_dic['Egress_upper'].append(bd['egress'][2])
        
        bestfitdic = eb.bestfit
        output_dic['Duration_bf'] = bestfitdic['duration']
        output_dic['Ingress_bf'] = bestfitdic['ingress']
        output_dic['Egress_bg'] = bestfitdic['egress']

        if ((bestfitdic['duration'] > bd['duration'][2]) 
            or (bestfitdic['duration'] < bd['duration'][1])
            or (bestfitdic['ingress'] > bd['ingress'][2])
            or (bestfitdic['ingress'] < bd['ingress'][1])
            or (bestfitdic['egress'] > bd['egress'][2])
            or (bestfitdic['egress'] < bd['egress'][1])):

            print("Houston we "+ '\u0336'.join('fucked up') + '\u0336' + 
                  " have a problem")

        output_dic['Duration'].append(r'${0}^{1}_{2}$'.format(
                bestfitdic['duration'],
                bd['duration'][1],
                bd['duration'][2],
                ))
        output_dic['Ingress'].append(r'${0}^{1}_{2}$'.format(
                bestfitdic['ingress'],
                bd['ingress'][1],
                bd['ingress'][2],
                ))
        output_dic['Duration'].append(r'${0}^{1}_{2}$'.format(
                bestfitdic['egress'],
                bd['egress'][1],
                bd['egress'][2],
                ))

    latex_dic = {
        'MainID':output_dic['MainID'],
        'ID': output_dic['ID'],
        'Spectral Type':output_dic['Spectral Type'],
        'Obs Start':output_dic['Obs Start'],
        'Duration':output_dic['Duration'],
        'Ingress':output_dic['Ingress'],
        'Egress':output_dic['Egress'],
    }
    df_latex = pd.DataFrame(latex_dic)

    with open('EclipseFits.tex', 'w') as f:
        f.write(df_latex.to_latex(index=False, escape=False))
        f.close()
    with open("EclipseFits.tex", 'f') as f:
        lines = f.readlines()
        f.close()
    
    lines.insert(3, "& & & (JD) & (min) & (min) & (min) \\\ \n")

    with open("EclipseFits.tex", 'w') as f:
        contents = "".join(lines)
        f.write(contents)
        f.close()

def main(iterations):
    print("---Running bootstrap wrapper function---")
    bootstrapwrapper(iterations)
    print("---Generating LaTeX table---")
    genTable()
    print("---Creating tower plot---")


                        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--iterations",
            help='Number of bootstrap fit iterations', 
            default=1000, type=int)
    parser.add_argument("--tower", 
            help="Generate tower plot", 
            default=False, action='store_true')
    args = parser.parse_args()

    #if args.tower:
    #    TowerPlot()
    #else:
    #    main(iterations=args.iterations)
    eb = WDEB('Gaia-DR2-6645284902019884928-NUV.csv')
    eb.fit(save=True)
    
