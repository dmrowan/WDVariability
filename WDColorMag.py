#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import argparse
import math
from matplotlib.patches import ConnectionPatch
from matplotlib.patheffects import withStroke
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import gaussian_kde
import WDutils

#Dom Rowan REU 2018

desc="""
WDColorMag: Produce color mag plot with shading based on best rank. 
Used to help identify interesting sources/confirm ranking.
"""

#Main Plotting function:
def main(mark, markgroup, ppuls, label):
    #Path assertions
    assert(os.path.isfile("Output/AllData.csv"))
    assert(os.path.isfile("/home/dmrowan/WhiteDwarfs/Catalogs/"
                          +"MainCatalog_reduced_simbad_asassn.csv"))

    #Read in dataframes
    df_rank = pd.read_csv("Output/AllData.csv")
    bigcatalog = pd.read_csv("/home/dmrowan/WhiteDwarfs/Catalogs/"
                             +"MainCatalog_reduced_simbad_asassn.csv")

    #Only looking at top 3000 sources
    df_rank_reduced = df_rank[:3001]
    #Read in ranks, sources
    no_gaia_data_counter = 0

    bp_rp_list = []
    absolute_mag_list = []
    rank_list = []
    source_list = []
    band_list = []
    for source, i in zip(df_rank_reduced['SourceName'], 
                         range(len(df_rank_reduced['SourceName']))):
        bigcatalog_idx = WDutils.catalog_match(source, bigcatalog)
        if len(bigcatalog_idx) == 0:
                print(source, "Not in catalog")
                no_gaia_data_counter += 1
        else:
            #Grab Gaia information 
            bigcatalog_idx = bigcatalog_idx[0]
        
            gaia_parallax = bigcatalog['gaia_parallax'][bigcatalog_idx]
            if str(gaia_parallax) == 'nan':
                no_gaia_data_counter += 1
            elif gaia_parallax <= 0:
                no_gaia_data_counter += 1
            else:
                #Calcualte bp-rp, absolute mag
                g_mag = bigcatalog['gaia_g_mean_mag'][bigcatalog_idx]
                bp_mag = bigcatalog['gaia_bp_mean_mag'][bigcatalog_idx]
                rp_mag = bigcatalog['gaia_rp_mean_mag'][bigcatalog_idx]

                absolute_mag_list.append(g_mag + 5*np.log10(gaia_parallax/100))
     
                bp_rp_list.append(bp_mag - rp_mag)

                rank_list.append(df_rank_reduced['BestRank'][i])
                source_list.append(source)
                band_list.append(df_rank_reduced['Band'][i])


    assert(len(bp_rp_list) == len(absolute_mag_list) 
           == len(rank_list) == len(source_list))

    #Plotting params
    df_plot = pd.DataFrame({'source':source_list, 
                            'band':band_list,
                            'rank':rank_list, 
                            'absolutemag':absolute_mag_list, 
                            'bp_rp':bp_rp_list})
    df_plot = df_plot.sort_values(by=['rank'])

    #Different plotting parameters. Using markgroup for paper plot generation
    if markgroup:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.tight_layout(rect=[.03, 0.1, 1, 1])
        fig.subplots_adjust(wspace=.125)
        ax1.scatter(df_plot['bp_rp'], df_plot['absolutemag'], 
                    c='gray', alpha=.5, label='_nolegend_')
        ax2.scatter(df_plot['bp_rp'], df_plot['absolutemag'], 
                    c='gray', alpha=.5, label='_nolegend_')
    else:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
        ax1.scatter(df_plot['bp_rp'], df_plot['absolutemag'], c='gray')

    ymin_val1 = 15
    ymax_val1 = 5.75
    xmin_val1 = -.85
    xmax_val1 = .65
    ax1.set_ylim(ymin=ymin_val1, ymax=ymax_val1)
    ax1.set_xlim(xmin=xmin_val1, xmax=xmax_val1)
    ax1 = WDutils.plotparams(ax1)
    ax1.set_xlabel("Gaia Color BP-RP (mag)", fontsize=20)
    ax1.set_ylabel("Absolute Gaia G (mag)", fontsize=20)

    myeffectw = withStroke(foreground="black", linewidth=2)
    txtkwargsw = dict(path_effects=[myeffectw])
    afont = {'fontname':'Keraleeyam'}

    #Second subplot params
    if markgroup:
        ax2.set_ylim(ymin=12.65, ymax=11.35)
        ax2.set_xlim(xmin=-.022, xmax=.16)
        ax2 = WDutils.plotparams(ax2)
        ax2.set_xlabel("Gaia Color BP-RP (mag)", fontsize=20)

    if mark is not None:
        #Determine if idx or source input:
        if mark.isdigit():
            mark = df_rank['SourceName'][int(mark)]

        if mark not in list(df_plot['source']):
            print("No gaia information on ", source)
        else:
            for i in range(len(df_plot['source'])):
                if df_plot['source'][i] == mark:
                    ii = i
                    break
            plt.plot([df_plot['bp_rp'][ii]], 
                     [df_plot['absolutemag'][ii]], 
                     'o', color='green', ms=12)
        
        plt.show()

    if markgroup:
        assert(os.path.isfile("/home/dmrowan/WhiteDwarfs/"
                              +"InterestingSources/IS.csv"))
        df_IS = pd.read_csv("/home/dmrowan/WhiteDwarfs/"
                            +"InterestingSources/IS.csv")

        if ppuls:
            assert(os.path.isfile("/home/dmrowan/WhiteDwarfs/"
                                  +"InterestingSources/IS_possible.csv"))
            df_ppuls = pd.read_csv("/home/dmrowan/WhiteDwarfs/"
                                   +"InterestingSources/IS_possible.csv")
            df_IS = df_IS.append(df_ppuls, ignore_index=True)
            
        for idx in range(len(df_IS['MainID'])):

            #Define color and labelnum
            labelnum = df_IS['labelnum'][idx]
            if df_IS['type'][idx] == 'Pulsator':
                typecolor = 'xkcd:red'
            elif df_IS['type'][idx] == 'KnownPulsator':
                typecolor = 'xkcd:violet'
            elif df_IS['type'][idx] == 'Eclipse':
                typecolor = 'xkcd:azure'
            else:
                assert(df_IS['type'][idx] == 'Possible')
                typecolor = 'xkcd:orange'

            #Iterate through points 
            for i in range(len(df_plot['source'])):
                if df_plot['source'][i] == df_IS['MainID'][idx]:
                    if ((df_plot['bp_rp'][i] > -.022)
                        and (df_plot['bp_rp'][i] < .16) 
                        and (df_plot['absolutemag'][i] < 12.65) 
                        and (df_plot['absolutemag'][i] > 11.35)):
                        #Include anotations w/ num label
                        if label:
                            ax2.text(df_plot['bp_rp'][i], 
                                     df_plot['absolutemag'][i], 
                                     str(labelnum),  ha='center', 
                                     va='center', 
                                     weight='normal', fontsize=12.5, 
                                     **txtkwargsw, zorder=4, color=typecolor,
                                     **afont)
                        else:
                            ax2.scatter(df_plot['bp_rp'][i],
                                        df_plot['absolutemag'][i], 
                                        c=typecolor, edgecolor='black',
                                        zorder=4, s=72, linewidths=2)
                    
                    if label:
                        ax1.text(df_plot['bp_rp'][i], 
                                 df_plot['absolutemag'][i], 
                                 str(labelnum),  ha='center', va='center', 
                                 weight='normal', fontsize=12.5, 
                                 **txtkwargsw, zorder=4, 
                                 **afont,color=typecolor)
                    else:
                        ax1.scatter(df_plot['bp_rp'][i], 
                                    df_plot['absolutemag'][i], c=typecolor,
                                    zorder=4, s=72, edgecolor='black', 
                                    linewidths=2)

        #Additional plotting params
        ax1.plot([-.022, -.022, .16, .16, -.022], 
                 [12.65, 11.35, 11.35, 12.65, 12.65], 
                 color='black', ls='--', lw=4)
        ax1.scatter(x=0, y=0, c='xkcd:red', s=100, 
                    edgecolor='black', linewidths=2,label='Pulsator')
        ax1.scatter(x=0, y=0, c='xkcd:violet', s=100, 
                    edgecolor='black', linewidths=2,label='KnownPulastor')
        ax1.scatter(x=0, y=0, c='xkcd:azure', s=100, 
                    edgecolor='black', linewidths=2,label='Eclipse')
        if ppuls:
            ax1.scatter(x=0, y=0, c='xkcd:orange', s=100, label='Possible')
        ax1.legend(loc=3, fontsize=15, scatterpoints=1)
     
        #Artist connecting point for zoom window
        con1 = ConnectionPatch(xyA=(.16,11.35), xyB=(-.022, 11.35), 
                               coordsA="data", coordsB="data", 
                               axesA=ax1, axesB=ax2, color='black', ls='--')
        con2 = ConnectionPatch(xyA=(.16, 12.65), xyB=(-.022, 12.65), 
                               coordsA="data", coordsB="data", 
                               axesA=ax1, axesB=ax2, color='black', ls='--')
        ax1.add_artist(con1)
        ax1.add_artist(con2)

        if ppuls:
            fig.savefig("/home/dmrowan/WhiteDwarfs/"
                        +"InterestingSources/CMDplot_ppuls.pdf")
        else:
            if label:
                fig.savefig("/home/dmrowan/WhiteDwarfs/"
                            +"InterestingSources/CMDplot.pdf")
            else:
                fig.savefig("/home/dmrowan/WhiteDwarfs/"
                            +"InterestingSources/CMDplot_nolabel.pdf")
        

def gaiacmd(gray, wd_only):
    #Path assertions
    assert(os.path.isfile("GaiaQuery.csv"))
    #Read in dataframes
    df = pd.read_csv("GaiaQuery.csv")

    absolute_mag_list = []
    bp_rp_list = []
    for i in range(len(df['phot_g_mean_mag'])):
        gmag = df['phot_g_mean_mag'][i]
        parallax = df['parallax'][i]
        color = df['bp_rp'][i]
        if math.isnan(gmag) or math.isnan(parallax) or math.isnan(color):
            continue
        else:
            absolute_mag_list.append(gmag + 5*np.log10(parallax/100))
            bp_rp_list.append(color)

    assert(len(bp_rp_list) == len(absolute_mag_list))

    #WD cut line (from JJs poster)
    xvals = np.arange(-1, 5.3, .1)
    yvals = [ 4*x + 9 for x in xvals ]

    absolute_mag_other = []
    bp_rp_other = []
    WD_mag = []
    WD_bp_rp = []
    for i in range(len(absolute_mag_list)):
        mag = absolute_mag_list[i]
        color = bp_rp_list[i]
        if mag > 4*color + 9:
            WD_mag.append(mag)
            WD_bp_rp.append(color)
        else:
            absolute_mag_other.append(mag)
            bp_rp_other.append(color)


    if not gray:
        colormag = np.vstack([WD_bp_rp, WD_mag])
        z = gaussian_kde(colormag)(colormag)

    #Plotting params
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.tight_layout(rect=[.08, 0.1, 1, 1])
    ax.scatter(bp_rp_other, absolute_mag_other,
                c='gray', alpha=.5, label='Non-WDs', edgecolor='')
    if not gray:
        ax.scatter(WD_bp_rp, WD_mag, c=z,label='Gaia WDs', edgecolor='')
        ax.plot(xvals, yvals, c='black', ls='--', label='WD Cut')
        ax.legend(loc=1, fontsize=25, edgecolor='black')
    else:
        ax.scatter(WD_bp_rp, WD_mag, c='gray', alpha=.5)

    ax = WDutils.plotparams(ax)
    ax.set_xlabel("Gaia Color BP-RP (mag)", fontsize=30)
    ax.set_ylabel("Absolute Gaia "+ r'$G$'+" (mag)", fontsize=30)

    #plt.legend.get_frame().set_linewidth(1.4)

    if not gray:
        fig.savefig("GaiaCMD.png")
    else:
        fig.savefig("GaiaCMD_gray.png")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
            "--mark", 
            help="Input idx of a source (in AllData) or name and mark"
            , default=None, type=str)
    parser.add_argument(
            "--markgroup", 
            help="Using this to make plots for paper", 
            default=False, action='store_true')
    parser.add_argument(
            "--ppuls", 
            help="Generate plot using possible pulsators", 
            default=False, action='store_true')
    parser.add_argument(
            "--l", 
            help="Don't use lable numbers", 
            default=True, action='store_false')
    args= parser.parse_args()

    main(
            args.mark, 
            args.markgroup, 
            args.ppuls,
            args.l,
        )
