#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import WDutils
from progressbar import ProgressBar
#Dom Rowan REU 2018

desc="""
WDAssign: Iterate through interesting sources and generate CSV 
for use in plot creation. Also assign number for use in labeling.
"""

def sigfiground_int(val, sigma):
    if str(val) == 'nan':
        return ""
    s = len(str(int(sigma)))
    s = s-1
    output_val = round(val, -s)
    output_sigma = round(sigma, -s)
    stringoutput = str(output_val) + r'$\pm$' + str(output_sigma)
    return(stringoutput)

def sigfiground_float(val, sigma):
    if str(val) == 'nan':
        return ""
    counter = 0
    for character in str(sigma):
        if character == '.':
            counter=0
        elif character == '0':
            counter += 1
        else:
            break
    s = counter + 1
    output_val = round(val, s)
    output_sigma = round(sigma, s)
    stringoutput = str(output_val) + r'$\pm$' + str(output_sigma)
    return(stringoutput)


def main(ppuls):
    #Path assertions 
    prefix = '/home/dmrowan/WhiteDwarfs/'
    bigcatalog_path = prefix+"Catalogs/MainCatalog_reduced_simbad_asassn.csv"
    sigmamagpath_NUV = prefix+"GalexData_run7/Catalog/SigmaMag_NUV.csv"
    sigmamagpath_FUV = prefix+"GalexData_run7/Catalog/SigmaMag_FUV.csv"
    percentilepath_NUV = prefix+"GalexData_run7/Catalog/magpercentiles_NUV.csv"
    percentilepath_FUV = prefix+"GalexData_run7/Catalog/magpercentiles_FUV.csv"
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
        bigcatalog_idx = WDutils.catalog_match(name, bigcatalog)
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

    ra_list_round = [ round(val,5) for val in ra_list ]
    dec_list_round = [ round(val,5) for val in dec_list ]
    g_list = [ round(val, 2) for val in g_list ]
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

    #Get false alarm probabilities
    fap_list = []
    assert(os.path.isfile("/home/dmrowan/WhiteDwarfs/"
                         +"GalexData_run7/Output/AllData.csv"))
    df_fap = pd.read_csv("/home/dmrowan/WhiteDwarfs/"
                         +"GalexData_run7/Output/AllData.csv")
    for name in interestingsources:
        if name == '2MASS-J03030835+0054438':
            fap_list.append(9.29244e-8)
        elif name == 'WD-J0855+0635':
            fap_list.append(9.28808e-11)
        else:
            ii = np.where(df_fap['SourceName'] == name)[0]
            if len(ii) == 0:
                print("Something wrong with ", name)
                return
            elif len(ii) == 1:
                ii = ii[0]
                fap_list.append(df_fap['False Alarm Prob.'][ii])
            else:
                temp_fap_list = []
                for i in ii:
                    temp_fap_list.append(df_fap['False Alarm Prob.'][i])
                fap_list.append(min(temp_fap_list))

    #fap_list = [ "{0:.3g}".format(fap) for fap in fap_list ]
    fap_list = [ round(np.log10(f), 3) for f in fap_list ] 

    spectral_type = []
    for name in interestingsources:
        if name in ['Gaia-DR2-5158584274609992320',
                               'SDSS-J102106.69+082724.8',
                               'SDSS-J123654.96+170918.7',
                               'Gaia-DR2-1476652053203050368']:
            spectral_type.append("DBV")
        elif name in ['WD 1104+656', 
                      '2MASS-J12233961-0056311',
                      'GALEX-2667197544394657586',
                      '2MASS-J03030835+0054438']:
            spectral_type.append("DA+M")
        elif name in ['Gaia-DR2-1009242620785138688',
                      'Gaia-DR2-6645284902019884928',
                      'SDSS-J212232.58-061839.7',
                      'SDSS-J220823.66-011534.1',
                      'SDSS-J234829.09-092500.9']:
            spectral_type.append("D?+?")
        else:
            spectral_type.append("DAV")



    #Create df, sort values by ra, assign label number for use in plots
    df_output = pd.DataFrame({
            "MainID":interestingsources,
            "ra":ra_list_round, 
            "dec":dec_list_round, 
            "g":g_list, 
            "type":objecttypes,
            "simbad type":stypes_list,
            "metric_NUV":metric_NUV, 
            "m_ab_NUV":m_ab_list_NUV, 
            "sigma_m_NUV":sigma_list_NUV,
            "metric_FUV":metric_FUV, 
            "m_ab_FUV":m_ab_list_FUV, 
            "sigma_m_FUV":sigma_list_FUV,
            "Welch Stetson":ws_list,
            "FAP":fap_list,
            "Spectral Type":spectral_type,
            "ra_unrounded":ra_list,
            "dec_unrounded":dec_list,

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
            "Source":df["MainID"],
            "ID\#":df["labelnum"],
            "RA":df["ra"],
            "Dec.":df["dec"],
            "Gaia G":df["g"],
            #r'$c_{\sigma_{mag, NUV}}$': df['metric_NUV'],
            #r'$c_{\sigma_{mag, FUV}}$': df['metric_FUV'],
            #"Welch Stetson I":df["Welch Stetson"],
            "Type":df["type"],
            "Spectral Type":df["Spectral Type"],
            r'$\log_{10}(\rm {LS FAP})^{\alpha}$': df['FAP'],
            #"LS FAP":df["FAP"],
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
        elif df_output['Type'][i] == 'KnownPulsator':
            df_output.loc[i, 'Type'] = 'Known Pulsator'

    for i in range(len(df_output['Source'])):
        if df_output['Source'][i][:4] == 'Gaia':
            ra = df['ra_unrounded'][i]
            dec = df['dec_unrounded'][i]
            coordstring = WDutils.tohms(ra, dec, fancy=True)
            df_output.loc[i, 'Source'] = 'WD J'+coordstring
        else:
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

    idx_cbs = np.where(df_output['Source'] == 'CBS 130')[0][0]
    df_output.loc[idx_cbs, 'Gaia G'] = 16.56
    
    with open("IS_latextable.tex", 'w') as f:
        f.write(df_output.to_latex(index=False, escape=False))
        f.close()

    with open("IS_latextable.tex", 'r') as f:
        lines = f.readlines()
        f.close()
    
    lines.insert(3, "& & (deg) & (deg) & (mag) & & & \\\ \n")

    with open("IS_latextable.tex", 'w') as f:
        contents = "".join(lines)
        f.write(contents)
        f.close()

def latex_known():
    df_ref = pd.read_csv("references.csv")
    df_output = pd.DataFrame({
        "Source":df_ref["MainID"],
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
    df_output['Reported Pulsation Periods'] = df_ref['Pulsation Periods']

    ID = []
    df_IS = pd.read_csv("IS.csv")
    for name in df_output['Source']:
        idx_IS = np.where(df_IS['MainID'] == name)[0]
        idx_IS = idx_IS[0]
        ID.append(df_IS['labelnum'][idx_IS])
    df_output['ID\#'] = ID
    

    idx_eclipse = []
    for i in range(len(df_output['Source'])):
        if str(df_output['RA'][i]) == 'nan':
            idx_eclipse.append(i)
    df_output = df_output.drop(idx_eclipse)
    df_output = df_output.sort_values(by='RA')
    df_output = df_output.reset_index(drop=True)

    df_output = df_output[['Source', 'ID\#', 'Reference', 
                           'Reported Pulsation Periods']]

    with open("IS_latextable_kp.tex", 'w') as f:
        f.write(df_output.to_latex(index=False, escape=False))
        f.close()

    with open("IS_latextable_kp.tex", 'r') as f:
        lines = f.readlines()
        f.close()

    lines.insert(3, "&  &  & (s) \\\ \n")

    with open("IS_latextable_kp.tex", 'w') as f:
        contents = "".join(lines)
        f.write(contents)
        f.close()

#Match with gentile & produce latex table
def gentile_fits():
    assert(os.path.isfile("IS.csv"))
    assert(os.path.isfile("gentilematch.csv"))
    df = pd.read_csv("IS.csv")
    gentile = pd.read_csv("gentilematch.csv")
    TH = []
    gH = []
    cH = []
    THe = []
    gHe = []
    cHe = []
    bestatm = []
    bestT = []
    bestlogg = []
    for i in range(len(df['MainID'])):
        idx = np.where(gentile['MainID'] == df['MainID'][i])[0]
        if len(idx) != 0:
            idx = idx[0]
            TH.append(gentile['Teff'][idx])
            gH.append(gentile['log_g'][idx])
            cH.append(gentile['chi2'][idx])
            THe.append(gentile['Teff_He'][idx])
            gHe.append(gentile['log_g_He'][idx])
            cHe.append(gentile['chisq_He'][idx])
            #Use He models for those in DBV region
            if df['MainID'][i] in ['Gaia-DR2-5158584274609992320',
                                   'SDSS-J102106.69+082724.8',
                                   'SDSS-J123654.96+170918.7',
                                   'Gaia-DR2-1476652053203050368']:
                bestatm.append('He')
                temp = gentile['Teff_He'][idx]
                temp_sigma = gentile['eTeff_He'][idx]
                #temp_string = sigfiground_int(temp, temp_sigma)
                temp = round(temp, -2)
                temp_sigma = round(temp_sigma, -2)
                temp_string = str(temp) + r'$\pm$' + str(temp_sigma)
                bestT.append(temp_string)

                gval = gentile['log_g_He'][idx]
                gval_sigma = gentile['elog_g_He'][idx]
                gval = '{:.2f}'.format(round(gval, 2)) 
                gval_sigma = '{:.2f}'.format(round(gval_sigma, 2)) 
                logg_string = gval + r'$\pm$' + gval_sigma
                #logg_string = sigfiground_float(gval, gval_sigma)
                bestlogg.append(logg_string)
            else:
                bestatm.append('H')
                temp = gentile['Teff'][idx]
                temp_sigma = gentile['eTeff'][idx]
                #temp_string = sigfiground_int(temp, temp_sigma)
                temp = round(temp, -2)
                temp_sigma = round(temp_sigma, -2)
                temp_string = str(temp) + r'$\pm$' + str(temp_sigma)
                bestT.append(temp_string)

                gval = gentile['log_g'][idx]
                gval_sigma = gentile['elog_g'][idx]
                #logg_string = sigfiground_float(gval, gval_sigma)
                gval = '{:.2f}'.format(round(gval, 2)) 
                gval_sigma = '{:.2f}'.format(round(gval_sigma, 2)) 
                logg_string = gval + r'$\pm$' + gval_sigma
                bestlogg.append(logg_string)

        else:
            TH.append("")
            gH.append("")
            cH.append("")
            THe.append("")
            gHe.append("")
            cHe.append("")
            bestatm.append("")
            bestT.append("")
            bestlogg.append("")


    df['Teff_H'] = TH
    df['log_g_H'] = gH
    df['Chi2_H'] = cH
    df['Teff_He'] = THe
    df['log_g_He'] = gHe
    df['Chi2_He'] = cHe
    df['Teff'] = bestT
    df['logg'] = bestlogg
    df['atm'] = bestatm
    df.to_csv("IS.csv", index=False)

    #Generate Latex Table
    dic_latex = {
            #'MainID':df['MainID'],
            'ID':df['labelnum'],
            'Model':df['atm'],
            r'$T_{\rm{eff}}$':df['Teff'],
            r'$\log\rm{g}$':df['logg'],
            #r'$\chi^2$':df['Chi2'],
        }

    df_latex = pd.DataFrame(dic_latex)
    idx_drop = []
    for i in range(len(df_latex['ID'])):
        if (str(df_latex[r'$T_{\rm{eff}}$'][i])[:3] == 'nan'
            or str(df_latex[r'$T_{\rm{eff}}$'][i]) == ''):
            idx_drop.append(i)

    df_latex = df_latex.drop(index = idx_drop)
    df_latex = df_latex.reset_index(drop=True)

    with open("GentileLatex.tex", 'w') as f:
        f.write(df_latex.to_latex(index=False, escape=False, 
                column_format='llrr'))
        f.close()

    with open("GentileLatex.tex", 'r') as f:
        lines = f.readlines()
        f.close()
    
    lines.insert(3, "& &  (K) & (g/cm"+r'$^2$'+") \\\ \n")

    with open("GentileLatex.tex", 'w') as f:
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
    parser.add_argument("--gentile", 
            help="Cross-match gentile fits",
            default=False, action='store_true')
    args= parser.parse_args()

    if args.latex:
        latextable()
    elif args.kp:
        latex_known()
    elif args.gentile:
        gentile_fits()
    else:
        main(ppuls=args.ppuls)
