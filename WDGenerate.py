#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
import WDAssign
import WDPlot_eclipse
import WDPlot_pulsator
import WDColorMag
import WDsigmamag
import WDlatexcommands
import WDmaghist
#Dom Rowan REU 2018

desc="""
WDGenerate: Generate all plots, tables, for paper
"""

def main(eclipsegenerate, pulsatorgenerate):
    #Path assertions 
    assert(os.getcwd() == '/home/dmrowan/WhiteDwarfs')
    assert(os.path.isdir('InterestingSources'))
    assert(os.path.isdir('GalexData_run6'))

    #Run WDAssign
    os.chdir('InterestingSources')
    WDAssign.main(False)
    WDAssign.latextable()

    #Run WDPlot_eclipse.py
    WDPlot_eclipse.main(eclipsegenerate)

    #Run WDPlot_pulsator.py
    WDPlot_pulsator.main(pulsatorgenerate)

    #Run WDmaghist.py
    WDmaghist.main()
    
    #Run WDColorMag.py
    os.chdir('../GalexData_run6')
    WDColorMag.main(False, False, False,None, True, False)

    #Run WDsigmamag.py
    WDsigmamag.percentile(True)

    #Run WDlatexcommands.py
    os.chdir('../')
    assert(os.getcwd() == '/home/dmrowan/WhiteDwarfs')
    WDlatexcommands.main(False)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
            "--generate_eclipse_info", 
            help="Generate eclipse information for WDPlot_eclipse.py", 
            default=False, action='store_true')
    parser.add_argument(
            "--generate_pulsator_info", 
            help="Generate pulsator information for WD_plot_pulsator.py",
            default=False, action='store_true')
    args= parser.parse_args()

    main(eclipsegenerate=args.generate_eclipse_info,
         pulsatorgenerate=args.generate_pulsator_info,
    )
