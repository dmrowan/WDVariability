#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
import pandas as pd
import numpy as np
from WDranker_2 import catalog_match
#Dom Rowan REU 2018

desc="""
WDAssign: Iterate through interesting sources and generate CSV 
for use in plot creation. Also assign number for use in labeling.
"""

def main(ppuls):
    #Path assertions 
    prefix = '/home/dmrowan/WhiteDwarfs/'
    bigcatalog_path = prefix+"Catalogs/MainCatalog_reduced_simbad_asassn.csv"
    sigmamagpath_NUV = prefix+"GalexData_run5/Catalog/SigmaMag_NUV.csv"
    sigmamagpath_FUV = prefix+"GalexData_run5/Catalog/SigmaMag_FUV.csv"
    percentilepath_NUV = prefix+"GalexData_run5/Catalog/magpercentiles_NUV.csv"
    percentilepath_FUV = prefix+"GalexData_run5/Catalog/magpercentiles_FUV.csv"
    assert(os.getcwd() == '/home/dmrowan/WhiteDwarfs/InterestingSources')
    assert(os.path.isdir("KnownPulsator"))
    assert(os.path.isdir("Pulsator"))
    assert(os.path.isdir("Eclipse"))
    assert(os.path.isdir("PossiblePulsator"))
    assert(os.path.isfile(bigcatalog_path))
    assert(os.path.isfile(sigmamagpath_NUV))
    assert(os.path.isfile(sigmamagpath_FUV))
    assert(os.path.isfile(percentilepath_NUV))
    assert(os.path.isfile(percentilepath_FUV))

    #Load in source names
    knownpulsators = os.listdir("KnownPulsator")
    newpulsators = os.listdir("Pulsator") 
    eclipses = os.listdir("Eclipse") 

    if ppuls:
        possiblepulsators = os.listdir("PossiblePulsator") 
        objecttypes = ["Possible"] * len(possiblepulsators)
        interestingsources = possiblepulsators
    else:
        interestingsources = knownpulsators + newpulsators + eclipses
        objecttypes = (["KnownPulsator"]*len(knownpulsators) 
                + ["Pulsator"]*len(newpulsators) 
                + ["Eclipse"]*len(eclipses)
            )

    #Need to query big catalog for g mag, ra, dec
    bigcatalog = pd.read_csv(bigcatalog_path)
    g_list = []
    ra_list = []
    dec_list = []
    stypes_list = []
    for name in interestingsources:
        bigcatalog_idx = catalog_match(name, bigcatalog)
        if len(bigcatalog_idx) == 0:
            print(name, "Not in catalog")
            g_list.append("")
            ra_list.append("")
            dec_list.append("")
        else:
            bigcatalog_idx = bigcatalog_idx[0]
            g_list.append(bigcatalog['gaia_g_mean_mag'][bigcatalog_idx])
            ra_list.append(bigcatalog['ra'][bigcatalog_idx])
            dec_list.append(bigcatalog['dec'][bigcatalog_idx])
            stypes_list.append(bigcatalog['SimbadTypes'][bigcatalog_idx])


    df_alldata = pd.read_csv("/home/dmrowan/WhiteDwarfs/"
                             +"GalexData_run6/Output/AllData.csv")
    ws_list = []
    for name in interestingsources:
        alldata_idx = np.where(df_alldata['SourceName'] == name)[0]
        assert(len(alldata_idx) != 0)
        ws_values = [ df_alldata['WS metric'][ii] for ii in alldata_idx ]
        for ii in alldata_idx:
            ws_values.append(df_alldata['WS metric'][ii])
        ws_list.append(round(max(ws_values)))

    ra_list = [ round(val,5) for val in ra_list ]
    dec_list = [ round(val,5) for val in dec_list ]
    g_list = [ round(val, 1) for val in g_list ]
    #Query sigmamag csv's go get sigma m, m_ab, and our variability metric NUV
    metric_NUV= []
    sigma_list_NUV = []
    m_ab_list_NUV = []
    df_sm_NUV = pd.read_csv(sigmamagpath_NUV)
    percentile_df_NUV = pd.read_csv(percentilepath_NUV)
    magbins_NUV = percentile_df_NUV['magbin']
    magbins_NUV = np.array(magbins_NUV)
    percentile50_NUV = percentile_df_NUV['median']
    for name in interestingsources:
        idx = np.where(df_sm_NUV['Source'] == name)[0]
        #Cases for no idx, one idx, two idx
        if len(idx) == 0:
            print(name, "NUV not in sigma mag")
            metric_NUV.append("")
            sigma_list_NUV.append("")
            m_ab_list_NUV.append("")
        elif len(idx) == 1:
            idx = idx[0]
            m_ab = df_sm_NUV['m_ab'][idx]
            sigma_m = df_sm_NUV['sigma_m'][idx]
            absvals = np.array(abs(m_ab-magbins_NUV))
            sigmamag_idx = np.where(absvals == min(absvals))[0]
            sigmafit_val = float(percentile50_NUV[sigmamag_idx])
            metric_NUV.append(sigma_m / sigmafit_val)
            m_ab_list_NUV.append(m_ab)
            sigma_list_NUV.append(sigma_m)

    #Query sigmamag csv's go get sigma m, m_ab, and our variability metric FUV
    metric_FUV= []
    sigma_list_FUV = []
    m_ab_list_FUV = []
    df_sm_FUV = pd.read_csv(sigmamagpath_FUV)
    percentile_df_FUV = pd.read_csv(percentilepath_FUV)
    magbins_FUV = percentile_df_FUV['magbin']
    magbins_FUV = np.array(magbins_FUV)
    percentile50_FUV = percentile_df_FUV['median']
    for name in interestingsources:
        idx = np.where(df_sm_FUV['Source'] == name)[0]
        #Cases for no idx, one idx, two idx
        if len(idx) == 0:
            print(name, "FUV not in sigma mag")
            metric_FUV.append("")
            sigma_list_FUV.append("")
            m_ab_list_FUV.append("")
        elif len(idx) == 1:
            idx = idx[0]
            m_ab = df_sm_FUV['m_ab'][idx]
            sigma_m = df_sm_FUV['sigma_m'][idx]
            absvals = np.array(abs(m_ab-magbins_FUV))
            sigmamag_idx = np.where(absvals == min(absvals))[0]
            sigmafit_val = float(percentile50_FUV[sigmamag_idx])
            metric_FUV.append(sigma_m / sigmafit_val)
            m_ab_list_FUV.append(m_ab)
            sigma_list_FUV.append(sigma_m)

    metric_FUV = [ round(val, 2) for val in metric_FUV ]
    metric_NUV = [ round(val, 2) for val in metric_NUV ]
    #Create df, sort values by ra, assign label number for use in plots
    df_output = pd.DataFrame({
            "MainID":interestingsources,
            "ra":ra_list, 
            "dec":dec_list, 
            "g":g_list, 
            "type":objecttypes,
            "simbad type":stypes_list,
            "metric_NUV":metric_NUV, 
            "m_ab_NUV":m_ab_list_NUV, 
            "sigma_m_NUV":sigma_list_NUV,
            "metric_FUV":metric_FUV, 
            "m_ab_FUV":m_ab_list_FUV, 
            "sigma_m_FUV":sigma_list_FUV,
            "Welch Stetson":ws_list
        })

    df_output = df_output.sort_values(by=["ra"])
    df_output['labelnum'] = list(range(len(interestingsources)))
    if ppuls:
        df_output.to_csv("IS_possible.csv", index=False)
    else:
        df_output.to_csv("IS.csv", index=False)

def latextable():
    assert(os.path.isfile("IS.csv"))
    #assert(os.path.isfile("references.csv"))
    df = pd.read_csv("IS.csv")
    df_ref = pd.read_csv("references.csv")
    df_output = pd.DataFrame({
            "MainID":df["MainID"],
            "ID Num":df["labelnum"],
            "RA":df["ra"],
            "DEC":df["dec"],
            "Gaia G":df["g"],
            r'$c_{\sigma_{mag, NUV}}$': df['metric_NUV'],
            r'$c_{\sigma_{mag, FUV}}$': df['metric_FUV'],
            #"Welch Stetson I":df["Welch Stetson"],
            "Type":df["type"],
        })

    """
    references = []
    for i in range(len(df_output['MainID'])):
        idx_ref = np.where(df_ref['MainID'] == df_output['MainID'][i])[0]
        if len(idx_ref) == 0:
            references.append("")
        else:
            idx_ref = idx_ref[0]
            authoryear = df_ref['Reference'][idx_ref]
            latexcitation = "\cite{"+authoryear+"}"
            references.append(latexcitation)
    """

    for i in range(len(df_output['Type'])):
        if df_output['Type'][i] == 'Pulsator':
            df_output.loc[i, 'Type'] = "New Pulsator"
    
    with open("IS_latextable.tex", 'w') as f:
        f.write(df_output.to_latex(index=False, escape=False))
        f.close()

    with open("IS_latextable.tex", 'r') as f:
        lines = f.readlines()
        f.close()
    
    lines.insert(3, "& & (Deg) & (Deg) & (mag) & & (mag) & (mag)  \\\ \n")

    with open("IS_latextable.tex", 'w') as f:
        contents = "".join(lines)
        f.write(contents)
        f.close()

def latex_known():
    df_ref = pd.read_csv("references.csv")
    df_output = pd.DataFrame({
        "MainID":df_ref["MainID"],
        "RA":df_ref["RA"],
        "DEC":df_ref["DEC"],
    })
    references = []
    for citation in df_ref['Reference']:
        if str(citation) == 'nan':
            references.append("")
        else:
            latexcitation = "\cite{"+citation+"}"
            references.append(latexcitation)

    df_output['Reference'] = references
    df_output['Pulsation Periods'] = df_ref['Pulsation Periods']

    idx_eclipse = []
    for i in range(len(df_output['MainID'])):
        if str(df_output['RA'][i]) == 'nan':
            idx_eclipse.append(i)
    df_output = df_output.drop(idx_eclipse)
    df_output = df_output.sort_values(by='RA')
    df_output = df_output.reset_index(drop=True)

    with open("IS_latextable_kp.tex", 'w') as f:
        f.write(df_output.to_latex(index=False, escape=False))
        f.close()

    with open("IS_latextable_kp.tex", 'r') as f:
        lines = f.readlines()
        f.close()

    lines.insert(3, "& (deg) & (deg) &  &  (s) \\\ \n")

    with open("IS_latextable_kp.tex", 'w') as f:
        contents = "".join(lines)
        f.write(contents)
        f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--ppuls", 
            help="Create table for possible pulsators", 
            default=False, action='store_true')
    parser.add_argument("--latex", 
            help="Generate latex table", 
            default=False, action='store_true')
    parser.add_argument("--kp", 
            help="Generate latex table for known pulsators",
            default=False, action='store_true')
    args= parser.parse_args()

    if args.latex:
        latextable()
    elif args.kp:
        latex_known()
    else:
        main(ppuls=args.ppuls)
