#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import matplotlib.gridspec as gs
from gPhoton import gphoton_utils
import WDutils
from matplotlib.patheffects import withStroke
import _pickle as pickle
from scipy import optimize
from scipy.stats import norm, chisquare
import _pickle as pickle
from scipy.optimize import curve_fit
from progressbar import ProgressBar
import matplotlib.image as mpimg
from PIL import Image

desc="""
WD_SED: Define a class for SEDs
"""

def blackbody(x, a, T):
    #Input x in m, T in K
    #Constants
    h = 6.626e-27
    c = 2.99793e10
    kb = 1.381e-16
    firstterm = (2*h*c**2) / (x**5)
    secondterm = (1) / (np.exp((h*c)/(x*kb*T))-1)
    return a*firstterm*secondterm



#SED class
class WDSED:
    def __init__(self, source=None, ID=None):
        assert(os.path.isfile("IS.csv"))
        df_IS = pd.read_csv('IS.csv')

        if source is None and ID is None:
            print("Input source or ID number")
        elif source is not None:
            assert(type(source) == str)
            self.source = source
            idx_IS = np.where(df_IS['MainID'] == self.source)[0]
            assert(len(idx_IS) == 1)
            idx_IS = idx_IS[0]
            self.objecttype = df_IS['type'][idx_IS]
            self.labelnum = df_IS['labelnum'][idx_IS]
        else:
            assert(ID is not None)
            self.labelnum = ID
            idx_IS = np.where(df_IS['labelnum'] == int(ID))[0]
            assert(len(idx_IS) == 1)
            idx_IS = idx_IS[0]
            self.objecttype = df_IS['type'][idx_IS]
            self.source = df_IS['MainID'][idx_IS]

        #originally from gentile fusillo fits
        self.TeffH = df_IS['Teff_H'][idx_IS]
        self.loggH = df_IS['log_g_H'][idx_IS]
        self.chi2H = df_IS['Chi2_H'][idx_IS]
        
        self.TeffHe = df_IS['Teff_He'][idx_IS]
        self.loggHe = df_IS['log_g_He'][idx_IS]
        self.chi2He = df_IS['Chi2_He'][idx_IS]

        if str(self.TeffH) == 'nan':
            print("No fit information for {}".format(self.source))
            self.Teff = 11000
            self.gentileexists = False
        else:
            self.gentileexists = True
            """
            if self.chi2H < self.chi2He:
                self.atmfit = "H"
                self.Teff = self.TeffH
                self.logg = self.loggH
                self.chi2 = self.chi2H
            else:
                self.atmfit = "He"
                self.Teff = self.TeffHe
                self.logg = self.loggHe
                self.chi2 = self.chi2He
            """
            self.atmfit = "H"
            self.Teff = self.TeffH
            self.logg = self.loggH
            self.chi2 = self.chi2H

        
        #Read in vosa sed info
        fpath = self.objecttype+"/"+self.source+"/sed.dat"
        if not os.path.isfile(fpath):
            print("No sed file found for {}".format(self.source))
            self.sedexists = False
        else:
            self.sedexists = True
            with open(fpath) as f:
                lines = f.readlines()

            #Initialize dictionary of all filters
            dic_all = {
                    'filter':[], 
                    'wavelength':[],
                    'flux':[],
                    'flux_err':[],
                    'uplim':[]
                }

            #Iterate through lines
            for line in lines:
                if line.split()[0] != '#' and len(line.split()) != 1:
                    if float(line.split()[4]) < 0:
                        continue
                    else:
                        dic_all['filter'].append(line.split()[0])
                        dic_all['wavelength'].append(float(line.split()[1]))
                        dic_all['flux'].append(float(line.split()[4]))
                        dic_all['flux_err'].append(float(line.split()[5]))
                        if 'uplim' in line.split():
                            dic_all['uplim'].append(1)
                        else:
                            dic_all['uplim'].append(0)
            
            df_all = pd.DataFrame(dic_all)
            #Insert multiplicative factor 
            self.mf = 10**19
            for i in range(len(df_all['flux'])):
                df_all.loc[i, 'flux'] = df_all['flux'][i] * self.mf
                df_all.loc[i, 'flux_err'] = df_all['flux_err'][i] * self.mf

            #Seperate into 3 seperate DFs
            idx_wise = []
            for idx in range(len(df_all['filter'])):
                if 'WISE' in df_all['filter'][idx]:
                    idx_wise.append(idx)

            df_wise = df_all.loc[idx_wise]
            df_wise = df_wise.reset_index(drop=True)
            df_nonIR = df_all.drop(index=idx_wise)
            df_nonIR = df_nonIR.reset_index(drop=True)

            #Assign to object attributes
            self.df_all = df_all
            self.df_wise = df_wise
            self.df_nonIR = df_nonIR


    #Add WISE information for sources not in VOSA
    def addWISE(self, W1=None, W1err=None, W2=None, W2err=None, 
                W3=None, W3err=None, W4=None, W4err=None):

        #Found at http://wise2.ipac.caltech.edu/docs/
        #release/allsky/expsup/sec4_4h.html
        F01 = 309.540
        F02 = 171.787
        F03= 31.674
        F04 = 8.363
        F0list = [F01, F02, F03, F04]

        l1 = 3.4*1e4
        l2 = 4.6*1e4
        l3 = 12*1e4
        l4 = 22*1e4
        llist = [l1, l2, l3, l4]

        Wlist = [W1, W2, W3, W4]
        Werr_list = [W1err, W2err, W3err, W4err]
        flux_densities = []
        errors = []

        for i in range(len(Wlist)):
            W = Wlist[i]
            Werr = Werr_list[i]
            F0 = F0list[i]
            l = llist[i]
            if W is not None:
                f = (3e-5)*F0*10**(-W/2.5)
                f = f / (l**2)
                flux_densities.append(f)
                if Werr is not None:
                    ferr = (3e-5)*F0*10**(-(W+Werr)/2.5)
                    ferr = ferr / (l**2)
                    errors.append(abs(ferr - f))
                else:
                    errors.append("")
            else:
                flux_densities.append("")
                errors.append("")


        #Add WISE information to df_wise, df_all
        df_add = pd.DataFrame({
                'filter':['WISE/WISE.W1', 'WISE/WISE.W2', 
                          'WISE/WISE.W3', 'WISE/WISE.W4'], 
                'wavelength':[33526.0, 46028.0, 115608.0, 220883.0], 
                'flux':flux_densities, 
                'flux_err':errors})
       
        idx_add_drop = []
        for ii in range(len(df_add['flux'])):
            if str(df_add['flux']) == 'nan':
                idx_add_drop.append(ii)

        self.df_add = df_add.drop(index=idx_add_drop)
        self.df_all = self.df_all.append(self.df_add, ignore_index=True)
        self.df_wise = self.df_wise.append(self.df_add, ignore_index=True)

    #Construct plot of SED
    def plotSED(self):
        if not self.sedexists:
            return
            savepath = ("/home/dmrowan/WhiteDwarfs/InterestingSources/"+
                       "SEDfits/"+self.source+".png")

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.errorbar(self.df_all['wavelength'], 
                    (self.df_all['flux'] / self.mf), 
                    yerr=(self.df_all['flux_err'] / self.mf),
                    marker='o', ecolor='gray',
                    ls='-',  label='VOSA data', color='xkcd:black')

        arrow = u'$\u2193$'
        myeffect = withStroke(foreground="k", linewidth=1.5)
        txtkwargs = dict(path_effects=[myeffect])
        myeffectw = withStroke(foreground="black", linewidth=2)
        txtkwargsw = dict(path_effects=[myeffectw])
        afont = {'fontname':'Keraleeyam'}

        for i in range(len(self.df_wise['flux'])):
            if self.df_wise['uplim'][i] == 1:
                ax.plot(self.df_wise['wavelength'][i]-1000, 
                           (self.df_wise['flux'][i] / self.mf), 
                           marker=arrow, label='_nolegend_', ls='',
                           zorder=3, markersize=20, color='xkcd:orangered')

        if self.objecttype == 'Pulsator':
            typecolor='xkcd:red'
        elif self.objecttype == 'KnownPulsator':
            typecolor='xkcd:violet'
        else:
            assert(self.objecttype == 'Eclipse')
            typecolor='xkcd:azure'
        ax.annotate(str(self.labelnum), xy=(.9, .9), 
                    xycoords='axes fraction', color=typecolor, 
                    fontsize=30, ha='center', va='center', 
                    **afont, **txtkwargsw)
        #Plot params
        ax.minorticks_on()
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(direction='in', which='both', labelsize=15)
        ax.tick_params('both', length=8, width=1.8, which='major')
        ax.tick_params('both', length=4, width=1, which='minor')

        ax.set_yscale('log')
        minf = min(self.df_all['flux'] / self.mf)
        maxf = max(self.df_all['flux'] / self.mf)
        ax.set_ylim(ymin=10**(round(np.log10(minf), 2)-.5),
                    ymax=10**(round(np.log10(maxf), 2)+.5))

        #mfpower = str(int(np.log10(self.mf)))
        #mflabel = r'$\times10^{-'+mfpower+'}$'
        #fig.text(.01, .95, mflabel, fontsize=20)
        ax.set_ylabel('Flux ('+r'erg/s/cm$^2$/Å)', fontsize=30)
        ax.set_xlabel('Wavelength (Å)', fontsize=30)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)

        #Save fig info
        self.SEDfig = fig
        self.SEDax = ax

    #Construct BB fit
    def BB(self):
        if not self.sedexists:
            return 
        self.plotSED()
        #Wavlength should be in cm
        wavelength_cm = self.df_nonIR['wavelength'] * 1e-8
        wavelength_cm_all = self.df_all['wavelength'] * 1e-8
        flux = self.df_nonIR['flux']
        #Initial conditions and bounds
        if self.source == 'US-1639':
            print(self.Teff)
            self.Teff = self.Teff + 3000
        p0 = [5e-13, self.Teff]
        if not self.gentileexists:
            bounds = ([5e-17, 8000], [1e-11, 15000])
        else:
            bounds = ([5e-17, self.Teff-400], [1e-11, self.Teff+400])
        #Scipy curve fit
        popt, pcov = curve_fit(blackbody, wavelength_cm, flux, 
                               p0=p0, bounds=bounds)
        if self.source == 'US-1639':
            print(popt)
        #Plot fit
        xvals = np.linspace(min(wavelength_cm_all), 
                            max(wavelength_cm_all), 250)
        fitvals = blackbody(xvals, *popt)

        ax = self.SEDax
        fig = self.SEDfig
        xvals_plot = [ x * 1e8 for x in xvals ] 
        ax.plot(xvals_plot, (fitvals / self.mf), color='purple', 
                label='_nolegend_', lw=4)
        plt.tight_layout()
        self.SEDfig = fig
        self.SEDax = ax

        self.popt = popt

    def showPlot(self, save=False, overwrite=False):
        if not self.sedexists:
            return

        savepath = ("/home/dmrowan/WhiteDwarfs/InterestingSources/"+
                   "SEDfits/"+self.source+".png")
        if os.path.isfile(savepath) and overwrite==False:
            return
        fig = self.SEDfig
        ax = self.SEDax
        if save:
            fig.savefig(savepath)
        else:
            plt.show()

    #Quantify IR excess
    def IRexcess(self):
        self.BB()
        if not self.sedexists:
            return
        dfIR = self.df_wise
        dic_excess = {
                'MainID':[self.source],
                'ID':[self.labelnum],
                'W1 excess':[""],
                'W2 excess':[""],
                'W3 excess':[""],
                'W4 excess':[""],
            }

        for i in range(len(dfIR['wavelength'])):
            xval = dfIR['wavelength'][i] * 1e-8
            fitval = blackbody(xval, *self.popt)
            f = dfIR['flux'][i]
            ferr =dfIR['flux_err'][i]
            filtername = dfIR['filter'][i]
            if ferr != 0:
                exc = f - fitval
                exc = exc / ferr
                if 'W1' in filtername: 
                    dic_excess['W1 excess'] = exc
                elif 'W2' in filtername: 
                    dic_excess['W2 excess'] = exc
                elif 'W3' in filtername:
                    dic_excess['W3 excess'] = exc
                else:
                    dic_excess['W4 excess'] = exc

        df_excess = pd.DataFrame(dic_excess)
        return(df_excess)

    #Access Wise image
    def FindWISE(self):
        if os.path.isfile("WISEimages/"+self.source+"_wise.png"):
            self.WISEimage = "WISEimages/{}_wise.png".format(self.source)
        else:
            self.WISEimage = None
        return self.WISEimage


def sedwrapper(overwrite=False):
    print("-----Running SED Wrapper-----")
    assert(os.path.isfile("IS.csv"))
    df_IS = pd.read_csv("IS.csv")
    #Initialize Excess dictionary
    for i in range(len(df_IS['MainID'])):
        sed = WDSED(ID = df_IS['labelnum'][i])
        sed.BB()
        sed.showPlot(save=True, overwrite=overwrite)
        if i == 0:
            df_excess = sed.IRexcess()
        else:
            new_df = sed.IRexcess()
            df_excess = df_excess.append(new_df, ignore_index=True)
    
    print(df_excess)
    df_excess.to_csv("ExcessTable.csv", index=False)


def ExcessTower():
    assert(os.path.isfile('ExcessTable_comments.csv'))
    df_comments = pd.read_csv('ExcessTable_comments.csv')
    sed_list = []
    for i in range(len(df_comments['Excess'])):
        if df_comments['Excess'][i] == 1:
            sed = WDSED(ID=df_comments['ID'][i])
            sed.BB()
            sed_list.append(sed)

    nfigs = len(sed_list)
    figT = plt.figure(figsize=(10,3.5))
    gsT = gs.GridSpec(nfigs,1)
    gsT.update(hspace=.2, wspace=.15)
    plt.subplots_adjust(top=.8, right=.98, bottom=.2, left=.17)
    figT.text(.02, .5, 'Flux',
              va='center', rotation='vertical', fontsize=30)
    figT.text(.05, .5, r'(erg/s/cm$^2$/Å)', 
              va='center', rotation='vertical', fontsize=30)
    figT.text(.55, .07, 'Wavelength (Å)', fontsize=30,
              va='center', ha='center')

    arrow = u'$\u2193$'
    myeffectw = withStroke(foreground='black', linewidth=2)
    txtkwargsw = dict(path_effects=[myeffectw])
    afont = {'fontname':'Keraleeyam'}

    coords = []
    for y in range(0, nfigs):
        coords.append( (y,0) )
        #coords.append( (y,1) )
    idx_plot = 0
    for sed in sed_list:
        if idx_plot == len(sed_list) + 1:
            break
        plot_coords = coords[idx_plot]
        plt.subplot2grid((nfigs,1), plot_coords, colspan=1, rowspan=1)
        axT = plt.subplot(gsT[plot_coords])
        #Plot SED
        axT.errorbar(sed.df_nonIR['wavelength'], 
                    (sed.df_nonIR['flux'] / sed.mf), 
                    yerr=(sed.df_nonIR['flux_err'] / sed.mf),
                    marker='o', ecolor='gray',
                    ls='-',  label='_nolegend_', color='xkcd:black')

        #Wise upper limit plot arrows
        for i in range(len(sed.df_wise['flux'])):
            if sed.df_wise['uplim'][i] == 1:
                axT.plot(sed.df_wise['wavelength'][i]-1000, 
                           (sed.df_wise['flux'][i] / sed.mf), 
                           marker=arrow, label='_nolegend_', ls='',
                           zorder=3, markersize=20, color='xkcd:orangered')
            else:
                if i == 0:
                    label='Photometry'
                else:
                    label='_nolegend_'
                axT.errorbar(sed.df_wise['wavelength'][i], 
                            (sed.df_wise['flux'][i] / sed.mf), 
                            yerr=(sed.df_wise['flux_err'][i] / sed.mf),
                            marker='o', ecolor='gray', 
                            ls='-',
                            color='xkcd:black',
                            label=label)
        #Plot blackbody
        popt = sed.popt
        wavelength_cm_all = sed.df_all['wavelength'] * 1e-8
        xvals = np.linspace(min(wavelength_cm_all), 
                            max(wavelength_cm_all), 250)
        fitvals = blackbody(xvals, *popt)
        xvals_plot = [ x * 1e8 for x in xvals ] 
        axT.plot(xvals_plot, (fitvals / sed.mf), color='purple', 
                label='BB Template', lw=4)

        #Type annotation
        if sed.objecttype == 'Pulsator':
            typecolor='xkcd:red'
        elif sed.objecttype == 'KnownPulsator':
            typecolor='xkcd:violet'
        else:
            assert(sed.objecttype == 'Eclipse')
            typecolor='xkcd:azure'
        axT.annotate(str(sed.labelnum), xy=(.95, .85), 
                    xycoords='axes fraction', color=typecolor, 
                    fontsize=20, ha='center', va='center', 
                    **afont, **txtkwargsw)
        #Plot params
        axT.minorticks_on()
        axT.yaxis.set_ticks_position('both')
        axT.xaxis.set_ticks_position('both')
        axT.tick_params(direction='in', which='both', labelsize=15)
        axT.tick_params('both', length=8, width=1.8, which='major')
        axT.tick_params('both', length=4, width=1, which='minor')

        axT.set_yscale('log', subsy=[2, 3, 4, 5, 6, 7, 8, 9])
        minf = min(sed.df_all['flux'] / sed.mf)
        maxf = max(sed.df_all['flux'] / sed.mf)
        axT.set_ylim(ymin=10**(round(np.log10(minf), 2)-.75),
                    ymax=10**(round(np.log10(maxf), 2)+.75))

        #Set xaxis
        if sed.df_wise['uplim'][2] == 1:
            axT.set_xlim(
                    xmax = 1.2*sed.df_wise['wavelength'][1])
        else:
            axT.set_xlim(
                    xmax = 1.2*sed.df_wise['wavelength'][2])

        for axis in ['top', 'bottom', 'left', 'right']:
            axT.spines[axis].set_linewidth(1.5)

        axT.set_xlim(xmin = -1000)

        if idx_plot == 0:
            axT.legend(fontsize=15, edgecolor='black',
                       loc=(.65, .65))
                    

        idx_plot += 1

    figT.savefig("ExcessTower.pdf")

def WISEdoublefig():
    assert(os.path.isfile('ExcessTable_comments.csv'))
    df_comments = pd.read_csv('ExcessTable_comments.csv')
    sed_list = []
    for i in range(len(df_comments['Excess'])):
        if df_comments['Excess'][i] == 1:
            sed = WDSED(ID=df_comments['ID'][i])
            sed.BB()
            sed.FindWISE()
            if sed.WISEimage is not None:
                sed_list.append(sed)
            else:
                print(sed.source, "No WISE png")

    arrow = u'$\u2193$'
    myeffectw = withStroke(foreground='black', linewidth=2)
    txtkwargsw = dict(path_effects=[myeffectw])
    afont = {'fontname':'Keraleeyam'}
    for sed in sed_list:
        figD, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6))
        #Plot SED
        ax1.errorbar(sed.df_nonIR['wavelength'], 
                    (sed.df_nonIR['flux'] / sed.mf), 
                    yerr=(sed.df_nonIR['flux_err'] / sed.mf),
                    marker='o', ecolor='gray',
                    ls='-',  label='VOSA data', color='xkcd:black')

        #Wise upper limit plot arrows
        for i in range(len(sed.df_wise['flux'])):
            if sed.df_wise['uplim'][i] == 1:
                ax1.plot(sed.df_wise['wavelength'][i]-1000, 
                           (sed.df_wise['flux'][i] / sed.mf), 
                           marker=arrow, label='_nolegend_', ls='',
                           zorder=3, markersize=20, color='xkcd:orangered')
            else:
                ax1.errorbar(sed.df_wise['wavelength'][i], 
                            (sed.df_wise['flux'][i] / sed.mf), 
                            yerr=(sed.df_wise['flux_err'][i] / sed.mf),
                            marker='o', ecolor='gray', 
                            ls='-', label='Wise Data',
                            color='xkcd:black')
        #Plot blackbody
        popt = sed.popt
        wavelength_cm_all = sed.df_all['wavelength'] * 1e-8
        xvals = np.linspace(min(wavelength_cm_all), 
                            max(wavelength_cm_all), 250)
        fitvals = blackbody(xvals, *popt)
        xvals_plot = [ x * 1e8 for x in xvals ] 
        ax1.plot(xvals_plot, (fitvals / sed.mf), color='purple', 
                label='_nolegend_', lw=4)

        #Type annotation
        if sed.objecttype == 'Pulsator':
            typecolor='xkcd:red'
        elif sed.objecttype == 'KnownPulsator':
            typecolor='xkcd:violet'
        else:
            assert(sed.objecttype == 'Eclipse')
            typecolor='xkcd:azure'
        ax1.annotate(str(sed.labelnum), xy=(.95, .85), 
                    xycoords='axes fraction', color=typecolor, 
                    fontsize=20, ha='center', va='center', 
                    **afont, **txtkwargsw)
        #Plot params
        ax1.minorticks_on()
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.tick_params(direction='in', which='both', labelsize=15)
        ax1.tick_params('both', length=8, width=1.8, which='major')
        ax1.tick_params('both', length=4, width=1, which='minor')

        ax1.set_yscale('log', subsy=[2, 3, 4, 5, 6, 7, 8, 9])
        minf = min(sed.df_all['flux'] / sed.mf)
        maxf = max(sed.df_all['flux'] / sed.mf)
        ax1.set_ylim(ymin=10**(round(np.log10(minf), 2)-.75),
                    ymax=10**(round(np.log10(maxf), 2)+.75))


        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(1.5)

        #PNG image:
        pngfile = sed.WISEimage
        img2 = mpimg.imread(pngfile)
        ax2.imshow(img2)
        ax2.axis('off')

        saveimagepath = "WISEimages/"+sed.source+"_double.pdf"
        figD.savefig(saveimagepath)


        
#Generate Latex table
#Columns: name, id, W1mag/err, W1excess, W2mag/err, W2excess
def ExcessLatex():
    assert(os.path.isfile('ExcessTable_comments.csv'))
    df_comments = pd.read_csv('ExcessTable_comments.csv')
    df_output = {'Source':[],
                 'ID\#':[],
                 r'$W1$':[],
                 r'$E_{W1}$':[],
                 r'$W2$':[],
                 r'$E_{W2}$':[],
                 r'$W3$':[],
                 r'$E_{W3}$':[],
                 r'$W1-W2$':[],
            }
    for i in range(len(df_comments['MainID'])):
        if df_comments['Excess'][i] == 1:
            df_output['Source'].append(df_comments['MainID'][i])
            df_output['ID\#'].append(df_comments['ID'][i])

            W1mag = str(round(df_comments['W1mag'][i],1))
            W1err = str(round(df_comments['W1sigma'][i],1))
            df_output[r'$W1$'].append(W1mag+r'$\pm$'+W1err)
            df_output[r'$E_{W1}$'].append(
                    round(df_comments['W1 excess'][i],1))

            W2mag = str(round(df_comments['W2mag'][i],1))
            W2err = str(round(df_comments['W2sigma'][i],1))
            if W2mag != 'nan':
                df_output[r'$W2$'].append(W2mag+r'$\pm$'+W2err)
                df_output[r'$E_{W2}$'].append(
                        round(df_comments['W2 excess'][i],1))
                df_output[r'$W1-W2$'].append(
                        str(round(float(W1mag)-float(W2mag),1))+r'$\pm$'+
                        str(round(float(W1err)+float(W2err),1)))
            else:
                df_output[r'$W2$'].append("-")
                df_output[r'$E_{W2}$'].append("-")
                df_output[r'$W1-W2$'].append("-")

            W3mag = str(round(df_comments['W3mag'][i],1))
            W3err = str(round(df_comments['W3sigma'][i],1))
            if W3mag != 'nan':
                df_output[r'$W3$'].append(W3mag+r'$\pm$'+W3err)
                df_output[r'$E_{W3}$'].append(
                        round(df_comments['W3 excess'][i],1))
            else:
                df_output[r'$W3$'].append("-")
                df_output[r'$E_{W3}$'].append("-")

    df_output = pd.DataFrame(df_output)

    for i in range(len(df_output['Source'])):
        name = df_output['Source'][i]
        nhyphens = len(np.where(np.array(list(name)) == '-')[0])
        if ( (name[:3] =='CBS') or
                (name[:2] == 'V*') or
                (name[:5] == 'GALEX') or
                (name[:2] == 'US' and name[:4] != 'USNO') or
                (name[:2] == 'GD')):
            df_output.loc[i, 'Source'] = name.replace('-', ' ')
        elif ( (name[:3] == 'KUV') or
                (name[:5] == '2MASS') or
                (name[:2] == 'WD') or
                (name[:2] == 'HS') or
                (name[:2] == 'HE') or
                (name[:4] == 'SDSS')):
            if '+' in name:
                df_output.loc[i, 'Source'] = name.replace(
                        '-', ' ').replace('+', r'$+$')
            else:
                df_output.loc[i, 'Source'] = name.replace(
                        '-', ' ', 1).replace('-', r'$-$')
        elif name[:4]=='USNO':
            df_output.loc[i, 'Source'] = name.replace(
                    '-', ' ', nhyphens-1)
        else:
            df_output.loc[i, 'Source'] = name

    with open('ExcessLatex.tex', 'w') as f:
        f.write(df_output.to_latex(index=False, escape=False))
        f.close()
    with open('ExcessLatex.tex', 'r') as f:
        lines = f.readlines()
        f.close()

    lines.insert(3, " & & (mag) & (mag) & (mag) & (mag) & (mag) & (mag) & (mag) \\\ \n")
    with open('ExcessLatex.tex', 'w') as f:
        contents = "".join(lines)
        f.write(contents)
        f.close()

def PSfits():
    assert(os.path.isdir("PSfits"))
    assert(os.path.isfile("IS.csv"))
    df = pd.read_csv("IS.csv")
    for filename in os.listdir("PSfits"):
        if filename.endswith(".png"):
            fig, ax = plt.subplots(1,1, figsize=(12,12))
            sourcename = filename[:-4]
            idx_IS = np.where(df['MainID'] == sourcename)[0]
            assert(len(idx_IS) == 1)
            idx_IS = idx_IS[0]
            labelnum = df['labelnum'][idx_IS]
            ax.imshow(mpimg.imread("PSfits/"+filename))
            myeffectw = withStroke(foreground='black', linewidth=2)
            txtkwargsw = dict(path_effects=[myeffectw])
            afont = {'fontname':'Keraleeyam'}
            ax.annotate(str(labelnum), xy=(.95, .85), 
                         xycoords='axes fraction', color='xkcd:red', 
                         fontsize=20, ha='center', va='center', 
                         **afont, **txtkwargsw)
            ax.axis('off')

            fig.savefig('PSfits/labeled/'+sourcename+".png")

def CutoutPlot():

    f1 = 'US-1639'
    path1 = 'PSfits/{}.png'.format(f1)
    wpath1 = 'WISEimages/{}_WISE.png'.format(f1)

    f2 = 'SDSS-J110505.94+583103.0'
    path2 = 'PSfits/{}.png'.format(f2)
    wpath2 = 'WISEimages/{}_WISE.png'.format(f2)

    assert(all([os.path.isfile(p) for p in [path1, path2]]))
    assert(all([os.path.isfile(p) for p in [wpath1, wpath2]]))
    assert(os.path.isfile('IS.csv'))
    df = pd.read_csv('IS.csv')

    myeffectw = withStroke(foreground='black', linewidth=2)
    txtkwargsw = dict(path_effects=[myeffectw])
    afont = {'fontname':'Keraleeyam'}

   
    
    fig, (ax0, ax1) = plt.subplots(2, 2, figsize=(100, 100))
    plt.subplots_adjust(hspace=0, wspace=-0.013)
    #plt.subplots_adjust(top=.98, right=.98, bottom=.05)
    filters = [r'$y$', r'$y$', '$W1$','$W1$']
    for f, p, ax, b in zip([f1, f2, f1, f2], 
                           [path1, path2, wpath1, wpath2], 
                           [ax0[0], ax1[0], ax0[1], ax1[1]],
                           filters):
        idx_IS = np.where(df['MainID'] == f)[0][0]
        labelnum = df['labelnum'][idx_IS]
        ax.imshow(mpimg.imread(p))
        minx, maxx = ax.get_xlim()
        center = (minx+maxx)/2
        ax.set_xlim(center-700, center+700)
        miny, maxy = ax.get_ylim()
        centery = (miny+maxy)/2 
        ax.set_ylim(centery+700, centery-700)
        ax.annotate(str(labelnum), xy=(.9, .9), 
                     xycoords='axes fraction', color='xkcd:red', 
                     fontsize=200, ha='center', va='center', 
                     **afont, **txtkwargsw)
        ax.annotate(b, xy=(.1, .9), 
                    xycoords='axes fraction', color='xkcd:red',
                    fontsize=200, ha='center', va='center',
                    **afont, **txtkwargsw)
        ax.tick_params('both', labeltop=False, labelbottom=False, 
                       labelleft=False, labelright=False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(20)
            ax.spines[axis].set_color('gray')

    fig.savefig('CutoutPlot.pdf')
    
def CutoutPlot2():
    f1 = 'US-1639'
    path1 = 'PSfits/{}.png'.format(f1)
    wpath1 = 'WISEimages/{}_WISE2.png'.format(f1)

    assert(all([os.path.isfile(p) for p in [path1, wpath1]]))
    assert(os.path.isfile('IS.csv'))
    df = pd.read_csv('IS.csv')

    myeffectw = withStroke(foreground='black', linewidth=2)
    txtkwargsw = dict(path_effects=[myeffectw])
    afont = {'fontname':'Keraleeyam'}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=-.013)
    plt.subplots_adjust(top=.98, right=.98, bottom=.05, left=.05)
    filters = [r'$y$', r'$W1$']
    
    idx_IS = np.where(df['MainID'] == f1)[0][0]
    labelnum = df['labelnum'][idx_IS]

    ax0.imshow(mpimg.imread(path1))
    minx, maxx = ax0.get_xlim()
    center = (minx+maxx)/2
    ax0.set_xlim(center-700, center+700)
    miny, maxy = ax0.get_ylim()
    centery = (miny+maxy)/2 
    ax0.set_ylim(centery+700, centery-700)

    ax1.imshow(mpimg.imread(wpath1))
    ax1.set_xlim(835-438, 835+438)
    ax1.set_ylim(450+438, 450-438)

    for ax, b in zip([ax0, ax1], [r'$y$', r'$W1$']):
        ax.annotate(str(labelnum), xy=(.9, .9), 
                     xycoords='axes fraction', color='xkcd:red', 
                     fontsize=25, ha='center', va='center', 
                     **afont, **txtkwargsw)
        ax.annotate(b, xy=(.1, .9), 
                    xycoords='axes fraction', color='xkcd:red',
                    fontsize=25, ha='center', va='center',
                    **afont, **txtkwargsw)
        ax.tick_params('both', labeltop=False, labelbottom=False, 
                       labelleft=False, labelright=False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(4)
            ax.spines[axis].set_color('gray')
        ax.tick_params('both', length=0, width=0, which='both')

    fig.savefig('CutoutPlot.pdf')




def main():
    #ExcessTower()
    #WISEdoublefig()
    #ExcessLatex()
    #PSfits()
    CutoutPlot2()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--source", help="source name", type=str)
    args = parser.parse_args()

    main()
    """
    sedobject = WDSED(ID=17)
    sedobject.plotSED()
    sedobject.BB()
    sedobject.showPlot(save=True)
    sedobject.IRexcess()
    """
    #ExcessLatex()

    #sedwrapper(overwrite=True)
