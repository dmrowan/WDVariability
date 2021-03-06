#!/usr/bin/env python
from __future__ import print_function, division
from astropy.stats import median_absolute_deviation
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from progressbar import ProgressBar
import WDutils
import matplotlib.gridspec as gs
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.font_manager

#Dom Rowan REU 2018

desc="""
This creates the sigma mag plot and generates the info used the WDranker. Pre-resiquite for WDranker
"""

def main():
    output_dic_NUV = {"Source":[], "m_ab":[], "sigma_m":[], "weight":[]}
    output_dic_FUV = {"Source":[], "m_ab":[], "sigma_m":[], "weight":[]}
    pbar = ProgressBar()
    for filename in pbar(os.listdir(os.getcwd())):
        if filename.endswith(".csv"):
            #Get the source name
            for i in range(len(filename)):
                character = filename[i]
                if character == 'c':
                    endidx=i-5
                    break
            
            if 'NUV' in filename:
                band='NUV'
            elif 'FUV' in filename:
                band='FUV'
            else:
                print("No band in filename", filename)

            source = filename[0:endidx]
            #Read in csv
            alldata = pd.read_csv(filename)

            #Drop rows with counts issues, badflags, low expt
            alldata = WDutils.df_fullreduce(alldata)

            #Get total magnitude average
            m_ab_all = np.nanmedian(alldata['mag_bgsub'])
            #sigma_all = np.nanstd( (alldata['mag_bgsub_err_1'] + alldata['mag_bgsub_err_2'])/2.0 )
            sigma_all = median_absolute_deviation(alldata['mag_bgsub'])
            #Add info to dictionary
            if (m_ab_all > 13) and (m_ab_all < 30):
                if band=='NUV':
                    output_dic_NUV['Source'].append(source)
                    output_dic_NUV['m_ab'].append(m_ab_all)
                    output_dic_NUV['sigma_m'].append(sigma_all)
                    output_dic_NUV['weight'].append(1)
                else:
                    assert(band=='FUV')
                    output_dic_FUV['Source'].append(source)
                    output_dic_FUV['m_ab'].append(m_ab_all)
                    output_dic_FUV['sigma_m'].append(sigma_all)
                    output_dic_FUV['weight'].append(1)

    #Make output df
    #Drop rows where there is no mean/sigma or 0 sigma
    output_df_NUV = pd.DataFrame(output_dic_NUV)
    dropnull_NUV = np.where( (output_df_NUV['m_ab'].isnull()) | (output_df_NUV['sigma_m'].isnull()) | (output_df_NUV['sigma_m']==0) )[0]
    print(len(dropnull_NUV))
    output_df_NUV = output_df_NUV.drop(index=dropnull_NUV)
    output_df_NUV.to_csv("Catalog/SigmaMag_NUV.csv", index=False)

    
    output_df_FUV = pd.DataFrame(output_dic_FUV)
    dropnull_FUV = np.where( (output_df_FUV['m_ab'].isnull()) | (output_df_FUV['sigma_m'].isnull()) | (output_df_FUV['sigma_m']==0) )[0]
    print(len(dropnull_FUV))
    output_df_FUV = output_df_FUV.drop(index=dropnull_FUV)
    output_df_FUV.to_csv("Catalog/SigmaMag_FUV.csv", index=False)

    """
    if showplot:
        #Get vectors
        mag = np.array(output_df['m_ab'])
        sigma = np.array(output_df['sigma_m'])
        sources = np.array(output_df['Source'])
        source_colors0 = []
        source_colors1 = []
        source_colors2 = []
        for source in sources:
            if len(source) != 0:
                if source[0] == 'S':
                    #For sdss sources use 1, 0, 0 Red
                    source_colors0.append('1')
                    source_colors1.append('0')
                    source_colors2.append('0')
                elif source[0] == 'G':
                    #For GAIA sources, use .3, .7, 0 green
                    source_colors0.append('.3')
                    source_colors1.append('.7')
                    source_colors2.append('0')
                elif source[0] == 'A':
                    #ATLAS sources use .3, .7, .9 blue
                    source_colors0.append('.3')
                    source_colors1.append('.7')
                    source_colors2.append('.9')
                else:
                    #Other sources use .5, .1, .7 purple
                    source_colors0.append('.5')
                    source_colors1.append('.1')
                    source_colors2.append('.7')

        rgba_colors_source = np.zeros((len(source_colors0), 4))
        rgba_colors_source[:,0] = source_colors0
        rgba_colors_source[:,1] = source_colors1
        rgba_colors_source[:,2] = source_colors2
        rgba_colors_source[:,3] = '1'

        plt.scatter(mag, sigma, color=rgba_colors_source, zorder=2)
        plt.ylim(ymin=0)
        plt.ylim(ymax=.10)
        plt.xlim(xmin=13)
        plt.xlim(xmax=24)
        plt.show()
    """

def percentile(showplot):
    assert(os.path.isfile("Catalog/SigmaMag_FUV.csv"))
    assert(os.path.isfile("Catalog/SigmaMag_NUV.csv"))
    input_df_FUV = pd.read_csv("Catalog/SigmaMag_FUV.csv")
    input_df_NUV = pd.read_csv("Catalog/SigmaMag_NUV.csv")

    #Read in able and sort by mag, drop mag > 21 FUV
    input_df_FUV_sorted = input_df_FUV.sort_values(by='m_ab')
    input_df_FUV_sorted = input_df_FUV_sorted.reset_index(drop=True)
    droprows_FUV = np.where(input_df_FUV_sorted['m_ab'] > 21.5)[0]
    input_df_FUV_reduced = input_df_FUV_sorted.drop(index=droprows_FUV)
    input_df_FUV_reduced = input_df_FUV_reduced.reset_index(drop=True)

    #Read in able and sort by mag, drop mag > 21 NUV
    input_df_NUV_sorted = input_df_NUV.sort_values(by='m_ab')
    input_df_NUV_sorted = input_df_NUV_sorted.reset_index(drop=True)
    droprows_NUV = np.where(input_df_NUV_sorted['m_ab'] > 21.5)[0]
    input_df_NUV_reduced = input_df_NUV_sorted.drop(index=droprows_NUV)
    input_df_NUV_reduced = input_df_NUV_reduced.reset_index(drop=True)

    #Bin by magnitude FUV
    breaks_FUV = []
    magbins_FUV = np.arange(13, 21.5, .5)
    mag_i_FUV = 0 
    for i in range(len(input_df_FUV_reduced['m_ab'])):
        if mag_i_FUV != len(magbins_FUV) - 1:
            if input_df_FUV_reduced['m_ab'][i] >= magbins_FUV[mag_i_FUV + 1]:
                breaks_FUV.append(i)
                mag_i_FUV += 1
    data_FUV = np.split(input_df_FUV_reduced, breaks_FUV)
    percentile50_FUV = []
    percentilelower_FUV = []
    percentileupper_FUV = []
    lowerbound = 5
    upperbound = 95
    for df in data_FUV:
        percentile50_FUV.append(np.percentile(df['sigma_m'], 50))
        percentilelower_FUV.append(np.percentile(df['sigma_m'],lowerbound))
        percentileupper_FUV.append(np.percentile(df['sigma_m'], upperbound))

    #Bin by magnitude NUV
    breaks_NUV = []
    magbins_NUV = np.arange(13, 21.5, .5)
    mag_i_NUV = 0 
    for i in range(len(input_df_NUV_reduced['m_ab'])):
        if mag_i_NUV != len(magbins_NUV) - 1:
            if input_df_NUV_reduced['m_ab'][i] >= magbins_NUV[mag_i_NUV + 1]:
                breaks_NUV.append(i)
                mag_i_NUV += 1
    data_NUV = np.split(input_df_NUV_reduced, breaks_NUV)
    percentile50_NUV = []
    percentilelower_NUV = []
    percentileupper_NUV = []
    for df in data_NUV:
        percentile50_NUV.append(np.percentile(df['sigma_m'], 50))
        percentilelower_NUV.append(np.percentile(df['sigma_m'],lowerbound))
        percentileupper_NUV.append(np.percentile(df['sigma_m'], upperbound))
    
    output_dic_FUV = {'magbin':magbins_FUV, 'lower':percentilelower_FUV, 'median':percentile50_FUV, 'upper':percentileupper_FUV}
    output_dic_NUV = {'magbin':magbins_NUV, 'lower':percentilelower_NUV, 'median':percentile50_NUV, 'upper':percentileupper_NUV}
    output_df_FUV = pd.DataFrame(output_dic_FUV)
    output_df_NUV = pd.DataFrame(output_dic_NUV)
    output_df_NUV.to_csv("Catalog/magpercentiles_NUV.csv")
    output_df_NUV.to_csv("Catalog/magpercentiles_FUV.csv")

    if showplot:
        assert(os.path.isfile("/home/dmrowan/WhiteDwarfs/InterestingSources/IS.csv"))
        df_IS = pd.read_csv("/home/dmrowan/WhiteDwarfs/InterestingSources/IS.csv")

        #fig, ax = plt.subplots(2,1, figsize=(12,12), sharex=True)
        #fig.tight_layout(rect=[.03, 0.03, 1, 1])
        fig = plt.figure(figsize=(16,12))
        fig.text(.02, .5, r'$\sigma_{mag}$ (mag)', va='center', rotation='vertical', fontsize=30)
        gs1 = gs.GridSpec(2,1)
        gs1.update(hspace=0)
        ax0 = plt.subplot(gs1[0])
        ax1 = plt.subplot(gs1[1])
        ax0 = WDutils.plotparams(ax0)
        ax0.scatter(input_df_NUV_sorted['m_ab'], input_df_NUV_sorted['sigma_m'], color='gray', s=2, zorder=1, label='_nolegend_', alpha=.5)

        myeffect = withStroke(foreground="k", linewidth=1.5)
        txtkwargs = dict(path_effects=[myeffect])
        myeffectw = withStroke(foreground="black", linewidth=2)
        txtkwargsw = dict(path_effects=[myeffectw])

        for i in range(len(df_IS['MainID'])):
            if df_IS['type'][i] == 'Pulsator':
                typecolor = 'xkcd:red'
            elif df_IS['type'][i] == 'KnownPulsator':
                typecolor = 'xkcd:violet'
            else:
                typecolor = 'xkcd:azure'
            
            labelnum = df_IS['labelnum'][i]

            if ((df_IS['sigma_m_NUV'][i] < .355) 
                    and (df_IS['m_ab_NUV'][i] < 21.25)):
                ax0.text(df_IS['m_ab_NUV'][i], df_IS['sigma_m_NUV'][i], 
                         str(labelnum),  horizontalalignment='center', 
                         verticalalignment='center', weight='normal', 
                         fontsize=12.5, **txtkwargsw, zorder=4, 
                         color=typecolor)


        ax0.scatter(x=0, y=0, c='xkcd:red', s=100, label='New Pulsator')
        ax0.scatter(x=0, y=0, c='xkcd:violet', s=100, label='Known Pulastor')
        ax0.scatter(x=0, y=0, c='xkcd:azure', s=100, label='Eclipse')
        ax0.legend(loc=2, fontsize=15, scatterpoints=1)
        #ax0.plot(magbins_NUV, percentilelower_NUV, color='blue', zorder=2, 
        #         linewidth=2, ls='-', 
        #         label=r'$\bar{\sigma}-s_{-1}$', alpha=.75)
        ax0.plot(magbins_NUV, percentile50_NUV, color='blue', zorder=2, 
                 linewidth=2, ls='-', 
                 label='Median RMS', alpha=.75)

        ax0.set_xlim(xmin=13.25, xmax=21.25)
        ax0.set_ylim(ymin=-.025, ymax=.355)
        ax0.legend(fontsize=15)
        ax0.text(15.2, .355-.035, "NUV", fontsize=20, verticalalignment='center', horizontalalignment='center')

        ax1.set_xlabel("UV Magnitude", fontsize=30)
        ax1.minorticks_on()
        ax1.tick_params(direction='in', which='both', labelsize=15)
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.tick_params('both', length=8, width=1.8, which='major')
        ax1.tick_params('both', length=4, width=1, which='minor')
        ax1.scatter(input_df_FUV_sorted['m_ab'], input_df_FUV_sorted['sigma_m'], color='gray', s=2, zorder=1, label='_nolegend_', alpha=.5)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(1.5)
        #ax1.plot(magbins_FUV, percentile50_FUV, color='blue', zorder=2, linewidth=2, label='50 percentile', alpha=.75)
        #ax[1].plot(magbins_FUV, percentileupper_FUV, color='blue', zorder=2, linewidth=2, ls='--', label=str(upperbound)+" percentile", alpha=.75)
        #ax1.plot(magbins_FUV, percentilelower_FUV, color='blue', 
        #         zorder=2, linewidth=2, ls='-', 
        #         label=r'$\bar{sigma}-1$sigma', alpha=.75)
        ax1.plot(magbins_FUV, percentile50_FUV, color='blue', 
                 zorder=2, linewidth=2, ls='-', 
                 label="Median RMS", alpha=.75)
        for i in range(len(df_IS['MainID'])):
            if df_IS['type'][i] == 'Pulsator':
                typecolor = 'xkcd:red'
            elif df_IS['type'][i] == 'KnownPulsator':
                typecolor = 'xkcd:violet'
            else:
                typecolor = 'xkcd:azure'
            
            labelnum = df_IS['labelnum'][i]

            if (df_IS['sigma_m_FUV'][i] < .405) and (df_IS['m_ab_FUV'][i] < 21.25):
                ax1.text(df_IS['m_ab_FUV'][i], df_IS['sigma_m_FUV'][i], str(labelnum),  horizontalalignment='center', verticalalignment='center', weight='normal', fontsize=12.5, **txtkwargsw, zorder=4, color=typecolor)

        ax1.scatter(x=0, y=0, c='xkcd:red', s=100, label='New Pulsator')
        ax1.scatter(x=0, y=0, c='xkcd:violet', s=100, label='Known Pulastor')
        ax1.scatter(x=0, y=0, c='xkcd:azure', s=100, label='Eclipse')
        ax1.set_xlim(xmin=13.25, xmax=21.25)
        ax1.set_ylim(ymin=-.025, ymax=.405)
        ax1.text(15.2, .405-.035, "FUV", fontsize=20, verticalalignment='center', horizontalalignment='center')

        plt.subplots_adjust(hspace=.1)
        fig.savefig("/home/dmrowan/WhiteDwarfs/InterestingSources/SigmaMagPlot.pdf")

def pick(csvname):
    sigmamag_path_NUV = "Catalog/SigmaMag_NUV.csv"
    sigmamag_percentile_path_NUV = "Catalog/magpercentiles_NUV.csv"
    sigmamag_path_FUV = "Catalog/SigmaMag_FUV.csv"
    sigmamag_percentile_path_FUV = "Catalog/magpercentiles_FUV.csv"
    assert(os.path.isfile(sigmamag_path_NUV))
    assert(os.path.isfile(sigmamag_percentile_path_NUV))
    assert(os.path.isfile(sigmamag_path_FUV))
    assert(os.path.isfile(sigmamag_percentile_path_FUV))

    assert(os.path.isfile(csvname))
    #Find source name from csvpath
    csvpath = csvname
    for i in range(len(csvpath)):
        character = csvpath[i]
        if character == 'c':
            endidx=i-5
            break
    source = csvpath[0:endidx]

    #Grab the band (also checks we have source csv)
    if csvpath[-7] == 'N':
        band = 'NUV'
        band_other = 'FUV'
    elif csvpath[-7] == 'F':
        band = 'FUV'
        band_other = 'NUV'
    else:
        print("Not source csv, skipping")
        return
    assert(band is not None)

    bandcolors = {'NUV':'red', 'FUV':'blue'}
    alldata = pd.read_csv(csvpath)
    ###Alldata table corrections###
    #Drop rows with > 10e10 in cps, cps_err, cps < .5
    idx_high_cps = np.where( (alldata['cps_bgsub'] > 10e10) |
            (alldata['cps_bgsub_err'] > 10e10) |
            (alldata['counts'] < 1) |
            (alldata['counts'] > 100000) |
            (alldata['flux_bgsub'] < 0) |
            (alldata['cps_bgsub'] < -10000) )[0]
    if len(idx_high_cps) != 0:
        alldata = alldata.drop(index = idx_high_cps)
        alldata = alldata.reset_index(drop=True)

    #Fix rows with incorrecct t_means by averaging t0 and t1
    idx_tmean_fix = np.where( (alldata['t_mean'] < 1) |
            (alldata['t_mean'] > alldata['t1']) |
            (np.isnan(alldata['t_mean'])) )[0]

    for idx in idx_tmean_fix:
        t0 = alldata['t0'][idx]
        t1 = alldata['t1'][idx]
        mean = (t1 + t0) / 2.0
        alldata['t_mean'][idx] = mean

    ###Apparent Magnitude### - could also be done using conversion from flux
    m_ab = np.nanmedian(alldata['mag_bgsub'])
    #sigma_mag_all = np.nanstd( (alldata['mag_bgsub_err_1']
            #+ alldata['mag_bgsub_err_2'])/2.0 )
    sigma_mag_all = np.nanstd(alldata['mag_bgsub'])
    magdic = {"mag":[m_ab], "sigma":[sigma_mag_all], "weight":[1]}

    #Read in mag percentile information
    if band == 'NUV':
        percentile_df = pd.read_csv(sigmamag_percentile_path_NUV)
    else:
        assert(band == 'FUV')
        percentile_df = pd.read_csv(sigmamag_percentile_path_FUV)


    fig, ax = plt.subplots(1,1,figsize=(8,4))
    if band == 'NUV':
        df_sigmamag = pd.read_csv(sigmamag_path_NUV)
    else:
        assert(band == 'FUV')
        df_sigmamag = pd.read_csv(sigmamag_path_FUV)
    
    
    #Pull values, weights
    allmags = df_sigmamag['m_ab']
    allsigma = df_sigmamag['sigma_m']
    df_alphas = df_sigmamag['weight']
    rgb_1 = np.zeros((len(df_alphas),4))
    rgb_1[:,3] = df_alphas
    #Create magnitude bins using np.digitize
    ax.scatter(allmags,allsigma,color=rgb_1, zorder=1, s=5)

    #Get information from magdic
    sourcemags = np.array(magdic['mag'])
    sourcesigmas =np.array(magdic['sigma'])
    sourcealphas = np.array(magdic['weight'])
    #Make lists for arrow points (above .3 sigma)
    arrow_mag = []
    arrow_sigma = []
    arrow_alpha = []
    idx_arrow = np.where(sourcesigmas > .3)[0]
    for idx in idx_arrow:
        arrow_mag.append(sourcemags[idx])
        arrow_sigma.append(.29)
        arrow_alpha.append(sourcealphas[idx])

    #Drop these indicies from the source arrays
    sourcemags = np.delete(sourcemags, idx_arrow)
    sourcesigmas = np.delete(sourcesigmas, idx_arrow)
    sourcealphas = np.delete(sourcealphas, idx_arrow)

    #Make color code information
    rgb_2 = np.zeros((len(sourcealphas), 4))
    rgb_2[:,0] = 1.0
    rgb_2[:,3] = sourcealphas

    rgb_arrow = np.zeros((len(arrow_alpha),4))
    rgb_arrow[:,0] = .3
    rgb_arrow[:,1] = .7
    rgb_arrow[:,2] = 1.0
    rgb_arrow[:,3] = arrow_alpha

    ax.scatter(sourcemags, sourcesigmas, color=rgb_2, zorder=2)
    ax.scatter(arrow_mag, arrow_sigma, color=rgb_arrow, 
                  marker="^", zorder=3)
    #ax.set_xlabel("NUV (mag)", fontsize=20)
    #ax.set_ylabel("RMS", fontsize=20)
    ax.set_ylim(ymin=-.02, ymax=.35)
    ax.set_xlim(xmin=13, xmax=21)
    ax.minorticks_on()
    ax.tick_params(direction='in', which='both', labelsize=15)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    fig.savefig("/home/dmrowan/WhiteDwarfs/InterestingSources/exampleRMS.pdf")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
            "--showplot", 
            help= "Display the plot after running", 
            default=False, action='store_true')
    parser.add_argument("--percentile", 
            help="Use existing csv to grab fit information, generate new csv", 
            default=False, action='store_true')
    args= parser.parse_args()

    if args.percentile:
        percentile(args.showplot)
    else:
        main()

    #pick('Gaia-DR2-1009242620785138688-FUV.csv')
