'''
Script for fitting D+, D0 and Ds+ invariant-mass spectra
run: python get_vn_vs_mass.py fitConfigFileName.yml centClass inputFileName.root outFileName.root
            [--refFileName][--isMC][--batch]
'''

import sys
import argparse
import ctypes
import numpy as np
import yaml
import os
from ROOT import TLatex, TFile, TCanvas, TLegend, TH1D, TH1F, TDatabasePDG, TGraphAsymmErrors # pylint: disable=import-error,no-name-in-module
from ROOT import gROOT, gPad, gInterpreter, kBlack, kRed, kAzure, kGray, kOrange, kGreen, kMagenta, kFullCircle, kFullSquare, kOpenCircle # pylint: disable=import-error,no-name-in-module
from flow_analysis_utils import get_centrality_bins, get_vnfitter_results, get_ep_vn, getD0ReflHistos, get_particle_info # pylint: disable=import-error,no-name-in-module
sys.path.append('../../..')
sys.path.append('../..')
gInterpreter.ProcessLine(f'#include "{os.getcwd().split('flow')[0] + 'flow'}/invmassfitter/InvMassFitter.cxx"')
gInterpreter.ProcessLine(f'#include "{os.getcwd().split('flow')[0] + 'flow'}/invmassfitter/VnVsMassFitter.cxx"')
from ROOT import InvMassFitter, VnVsMassFitter
from utils.StyleFormatter import SetGlobalStyle, SetObjectStyle, DivideCanvas
from utils.FitUtils import SingleGaus, DoubleGaus, DoublePeakSingleGaus, DoublePeakDoubleGaus, RebinHisto
from kde_producer import kde_producer

def get_vn_vs_mass(fitConfigFileName, centClass, inFileName,
                   outputdir, suffix, vn_method, batch):


    with open(fitConfigFileName, 'r', encoding='utf8') as ymlfitConfigFile:
        fitConfig = yaml.load(ymlfitConfigFile, yaml.FullLoader)

    gROOT.SetBatch(batch)
    SetGlobalStyle(padleftmargin=0.14, padbottommargin=0.12, padtopmargin=0.12, opttitle=1)
    cent, centMinMax = get_centrality_bins(centClass)

    # read global configuration
    ptMins = fitConfig['ptmins']
    ptMaxs = fitConfig['ptmaxs']
    ptLims = list(ptMins)
    nPtBins = len(ptMins)
    ptLims.append(ptMaxs[-1])
    ptBinsArr = np.asarray(ptLims, 'd')
    ptTit = '#it{p}_{T} (GeV/#it{c})'
    fixSigma = fitConfig['FixSigma']
    fixMean = fitConfig['FixMean']
    harmonic = fitConfig['harmonic']
    particleName = fitConfig['Dmeson']
    inclSecPeak = fitConfig['InclSecPeak']
    rebins = fitConfig['Rebin']
    if not isinstance(rebins, list):
        rebins = [rebins] * len(ptMins)
    massMins = fitConfig['MassMin']
    if not isinstance(massMins, list):
        massMins = [massMins] * len(ptMins)
    massMaxs = fitConfig['MassMax']
    if not isinstance(massMaxs, list):
        massMaxs = [massMaxs] * len(ptMins)
    useRefl = fitConfig.get('InclRefl')
    reflFile = fitConfig.get('ReflFile', None)

    # read fit configuration
    if not isinstance(fixSigma, list):
        fixSigma = [fixSigma for _ in ptMins]
    if not isinstance(fixMean, list):
        fixMean = [fixMean for _ in ptMins]
    SgnFuncStr = fitConfig['SgnFunc']
    if not isinstance(SgnFuncStr, list):
        SgnFuncStr = [SgnFuncStr] * nPtBins
    BkgFuncStr = fitConfig['BkgFunc']
    if not isinstance(BkgFuncStr, list):
        BkgFuncStr = [BkgFuncStr] * nPtBins
    BkgFuncVnStr = fitConfig['BkgFuncVn']
    if not isinstance(BkgFuncVnStr, list):
        BkgFuncVn = [BkgFuncVnStr] * nPtBins
    reflFuncStr = fitConfig['ReflFunc']

    # sanity check of fit configuration
    SgnFunc, BkgFunc, BkgFuncVn, degPol = [], [], [], []
    for iPt, (bkgStr, sgnStr, bkgVnStr) in enumerate(zip(BkgFuncStr, SgnFuncStr, BkgFuncVnStr)):
        degPol.append(-1)
        if bkgStr == 'kExpo':
            BkgFunc.append(InvMassFitter.kExpo)
        elif bkgStr == 'kLin':
            BkgFunc.append(InvMassFitter.kLin)
        elif bkgStr == 'kPol2':
            BkgFunc.append(InvMassFitter.kPol2)
        elif bkgStr == 'kPol3':
            BkgFunc.append(6)
            degPol[-1] = 3
            if len(ptMins) > 1 and inclSecPeak[iPt] == 1:
                print('ERROR: Pol3 and Pol4 fits work only with one bin if you have the secondary peak! Exit!')
                sys.exit()
        elif bkgStr == 'kPol4':
            BkgFunc.append(6)
            degPol[-1] = 4
            if len(ptMins) > 1 and inclSecPeak[iPt] == 1:
                print('ERROR: Pol3 and Pol4 fits work only with one bin if you have the secondary peak! Exit!')
                sys.exit()
        elif bkgStr == 'kPow':
            BkgFunc.append(InvMassFitter.kPow)
        elif bkgStr == 'kPowEx':
            BkgFunc.append(InvMassFitter.kPowEx)
        else:
            print('ERROR: only kExpo, kLin, kPol2, kPol3, kPol4, kPow, and kPowEx background functions supported! Exit')
            sys.exit()
        if bkgVnStr == 'kExpo':
            BkgFuncVn.append(InvMassFitter.kExpo)
        elif bkgVnStr == 'kLin':
            BkgFuncVn.append(InvMassFitter.kLin)
        elif bkgVnStr == 'kPol2':
            BkgFuncVn.append(InvMassFitter.kPol2)
        else:
            print('ERROR: only kExpo, kLin, and kPol2 background functions supported for vn! Exit')
            sys.exit()
        if sgnStr == 'kGaus':
            SgnFunc.append(InvMassFitter.kGaus)
        elif sgnStr == 'k2Gaus':
            SgnFunc.append(InvMassFitter.k2Gaus)
        elif sgnStr == 'k2GausSigmaRatioPar':
            SgnFunc.append(InvMassFitter.k2GausSigmaRatioPar)
        else:
            print('ERROR: only kGaus, k2Gaus and k2GausSigmaRatioPar signal functions supported! Exit!')
            sys.exit()

    KDEtemplatesFuncts = []
    if fitConfig.get('IncludeKDETempls'):
        KDEtemplates = [[None]*len(fitConfig['TemplsFlags']) for _ in range(len(ptMins))]
        for iPt in range(len(ptMins)):
            for iFlag, flag in enumerate(fitConfig['TemplsFlags']):
                if fitConfig.get('FromGrid'):
                    KDEtemplates[iPt][iFlag] = kde_producer(fitConfig['TemplsInputs'][iFlag],
                                                     'fM', ptMins[iPt], ptMaxs[iPt], flag, '',
                                                     fitConfig['TemplsTreeNames'][iFlag])
                elif fitConfig.get('FromFile'):
                    templFile = TFile.Open(f'{fitConfig["FromFile"]}', 'r')
                    KDEtemplates[iPt][iFlag] = templFile.Get(f'KDE_pt_{ptMins[iPt]}_{ptMaxs[iPt]}_flag{flag}')
                    templFile.Close()
                else:
                    print(f'ERROR: incorrect setting for including KDEs in fit! Exit!')
                    sys.exit()
        KDEtemplatesFuncts = [[KDE.GetFunction() for KDE in KDEtemplatesPt] for KDEtemplatesPt in KDEtemplates]
    
    # set particle configuration
    if particleName == 'Ds':
        _, massAxisTit, decay, massForFit = get_particle_info(particleName)
        massDplus = TDatabasePDG.Instance().GetParticle(411).Mass()
    else:
        _, massAxisTit, decay, massForFit = get_particle_info(particleName)

    # load histos
    infile = TFile.Open(inFileName)
    if not infile or not infile.IsOpen():
        print(f'ERROR: file "{inFileName}" cannot be opened! Exit!')
        sys.exit()
    hRel, hSig, hMassForRel, hMassForSig  = [], [], [], []
    hMass, hMassForFit, hVn, hVnForFit = [], [], [], []
    hMassIns, hMassOuts, hMassInsForFit, hMassOutsForFit = [], [], [], []
    fTotFuncMass, fTotFuncVn, fSgnFuncMass, fBkgFuncMass, fMassBkgRflFunc,fBkgFuncVn = [], [], [], [], [], []
    hMCSgn, hMCRefl = [], []
    fMassTemplFuncts = [[None]*len(fitConfig['TemplsFlags']) for _ in range(len(ptMins))] if fitConfig.get('IncludeKDETempls') else [] 
    fVnTemplFuncts = [[None]*len(fitConfig['TemplsFlags']) for _ in range(len(ptMins))] if fitConfig.get('IncludeKDETempls') else []
    hist_reso = infile.Get('hist_reso')
    hist_reso.SetDirectory(0)
    reso = hist_reso.GetBinContent(1)
    inclSecPeak = [inclSecPeak] * len(ptMins) if not isinstance(inclSecPeak, list) else inclSecPeak
    for iPt, (ptMin, ptMax, secPeak) in enumerate(zip(ptMins, ptMaxs, inclSecPeak)):
        if not vn_method == 'sp' and not vn_method == 'ep':
            print(f'loading: cent_bins{cent}/pt_bins{ptMin}_{ptMax}/hist_mass_cent{cent}_pt{ptMin}_{ptMax}')
            hMassIns.append(infile.Get(f'cent_bins{cent}/pt_bins{ptMin}_{ptMax}/hist_mass_inplane_cent{cent}_pt{ptMin}_{ptMax}'))
            hMassOuts.append(infile.Get(f'cent_bins{cent}/pt_bins{ptMin}_{ptMax}/hist_mass_outplane_cent{cent}_pt{ptMin}_{ptMax}'))
            hMassIns[iPt].SetDirectory(0)
            hMassOuts[iPt].SetDirectory(0)
        else:
            print(f'loading: cent_bins{cent}/pt_bins{ptMin}_{ptMax}/hist_vn_{vn_method}_pt{ptMin}_{ptMax}')
            hMass.append(infile.Get(f'cent_bins{cent}/pt_bins{ptMin}_{ptMax}/hist_mass_cent{cent}_pt{ptMin}_{ptMax}'))
            hVn.append(infile.Get(f'cent_bins{cent}/pt_bins{ptMin}_{ptMax}/hist_vn_{vn_method}_pt{ptMin}_{ptMax}'))
            hVn[iPt].SetDirectory(0)
            hMass[iPt].SetDirectory(0)
            SetObjectStyle(hMass[iPt], color=kBlack, markerstyle=kFullCircle)
            SetObjectStyle(hVn[iPt], color=kBlack, markerstyle=kFullCircle)    
    infile.Close()

    hSigmaToFix = None
    if fitConfig['FixSigmaRatio']:
        # load sigma of first gaussian
        infileSigma = TFile.Open(fitConfig['SigmaRatioFile'])
        if not infileSigma:
            print(f'ERROR: file "{infileSigma}" cannot be opened! Exit!')
            sys.exit()
        hSigmaToFix = infileSigma.Get('hRawYieldsSigma')
        hSigmaToFix.SetDirectory(0)
        if hSigmaToFix.GetNbinsX() != nPtBins:
            print('WARNING: Different number of bins for this analysis and histo for fix sigma')
        infileSigma.Close()
        # load sigma of second gaussian
        infileSigma2 = TFile.Open(fitConfig['SigmaRatioFile'])
        if not infileSigma2:
            print(f'ERROR: file "{infileSigma2}" cannot be opened! Exit!')
            sys.exit()
        hSigmaToFix2 = infileSigma2.Get('hRawYieldsSigma2')
        hSigmaToFix2.SetDirectory(0)
        if hSigmaToFix2.GetNbinsX() != nPtBins:
            print('WARNING: Different number of bins for this analysis and histo for fix sigma')
        infileSigma2.Close()

    # check reflections
    if useRefl and particleName == 'Dzero':
        if reflFile == '':
            reflFile = inFileName.replace('proj', 'proj_mc')
            useRefl, hMCSgn, hMCRefl = getD0ReflHistos(reflFile, ptMins, ptMaxs)
        else:
            useRefl, hMCSgn, hMCRefl = getD0ReflHistos(reflFile, ptMins, ptMaxs)
    else:
        useRefl = False

    # create histos for fit results
    if vn_method == 'sp' or vn_method == 'ep':
        hSigmaSimFit = TH1D('hSigmaSimFit', f';{ptTit};#sigma', nPtBins, ptBinsArr)
        hMeanSimFit = TH1D('hMeanSimFit', f';{ptTit};mean', nPtBins, ptBinsArr)
        hMeanSecPeakFitMass = TH1D('hMeanSecondPeakFitMass', f';{ptTit};mean second peak mass fit', nPtBins, ptBinsArr)
        hMeanSecPeakFitVn = TH1D('hMeanSecondPeakFitVn', f';{ptTit};mean second peak vn fit', nPtBins, ptBinsArr)
        hSigmaSecPeakFitMass = TH1D('hSigmaSecondPeakFitMass',
                                    f';{ptTit};width second peak mass fit', nPtBins, ptBinsArr)
        hSigmaSecPeakFitVn = TH1D('hSigmaSecondPeakFitVn', f';{ptTit};width second peak vn fit', nPtBins, ptBinsArr)
        hRawYieldsSimFit = TH1D('hRawYieldsSimFit', f';{ptTit};raw yield', nPtBins, ptBinsArr)
        hRawYieldsTrueSimFit = TH1D('hRawYieldsTrueSimFit', f';{ptTit};raw yield true', nPtBins, ptBinsArr)
        hRawYieldsSecPeakSimFit = TH1D('hRawYieldsSecondPeakSimFit',
                                       f';{ptTit};raw yield second peak', nPtBins, ptBinsArr)
        hRawYieldsSignificanceSimFit = TH1D('hRawYieldsSignificanceSimFit',
                                            f';{ptTit};significance', nPtBins, ptBinsArr)
        hRawYieldsSoverBSimFit = TH1D('hRawYieldsSoverBSimFit', f';{ptTit};S/B', nPtBins, ptBinsArr)
        hRedChi2SimFit = TH1D('hRedChi2SimFit', f';{ptTit};#chi^{{2}}/#it{{ndf}}', nPtBins, ptBinsArr)
        hProbSimFit = TH1D('hProbSimFit', f';{ptTit};prob', nPtBins, ptBinsArr)
        hRedChi2SBVnPrefit = TH1D('hRedChi2SBVnPrefit', f';{ptTit};#chi^{{2}}/#it{{ndf}}', nPtBins, ptBinsArr)
        hProbSBVnPrefit = TH1D('hProbSBVnPrefit', f';{ptTit};prob', nPtBins, ptBinsArr)
        hvnSimFit = TH1D('hvnSimFit',f';{ptTit};V2 ({vn_method})', nPtBins, ptBinsArr)

        SetObjectStyle(hSigmaSimFit, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hMeanSimFit, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hMeanSecPeakFitMass, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hSigmaSecPeakFitMass, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hMeanSecPeakFitVn, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hSigmaSecPeakFitVn, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hRawYieldsSimFit, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hRawYieldsTrueSimFit, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hRawYieldsSecPeakSimFit, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hRawYieldsSignificanceSimFit, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hRawYieldsSoverBSimFit, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hRedChi2SimFit, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hProbSimFit, color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hRedChi2SBVnPrefit, color=kRed, markerstyle=kFullSquare)
        SetObjectStyle(hProbSBVnPrefit, color=kRed, markerstyle=kFullSquare)
        SetObjectStyle(hvnSimFit, color=kBlack, markerstyle=kFullCircle)

    else:
        hRawYieldsIn = TH1D('hRawYieldsIn', f';{ptTit};raw yield in-plane', nPtBins, ptBinsArr)
        hRawYieldsOut = TH1D('hRawYieldsOut', f';{ptTit};raw yield out-of-plane', nPtBins, ptBinsArr)
        hSigmaIn = TH1D('hSigmaIn', f';{ptTit};#sigma in-plane', nPtBins, ptBinsArr)
        hSigmaOut = TH1D('hSigmaOut', f';{ptTit};#sigma out-of-plane', nPtBins, ptBinsArr)
        hMeanIn = TH1D('hMeanIn', f';{ptTit};mean in-plane', nPtBins, ptBinsArr)
        hMeanOut = TH1D('hMeanOut', f';{ptTit};mean out-of-plane', nPtBins, ptBinsArr)
        hRedChi2In = TH1D('hRedChi2In', f';{ptTit};#chi^{{2}}/#it{{ndf}} in-plane', nPtBins, ptBinsArr)
        hRedChi2Out = TH1D('hRedChi2Out', f';{ptTit};#chi^{{2}}/#it{{ndf}} out-of-plane', nPtBins, ptBinsArr)
        hProbIn = TH1D('hProbIn', f';{ptTit};prob in-plane', nPtBins, ptBinsArr)
        hProbOut = TH1D('hProbOut', f';{ptTit};prob out-of-plane', nPtBins, ptBinsArr)
        hRawYieldsSignificanceIn = TH1D('hRawYieldsSignificanceIn', f';{ptTit};significance in-plane',
                                        nPtBins, ptBinsArr)
        hRawYieldsSignificanceOut = TH1D('hRawYieldsSignificanceOut', f';{ptTit};significance out-of-plane',
                                         nPtBins, ptBinsArr)
        hRawYieldsSoverBIn = TH1D('hRawYieldsSoverBIn', f';{ptTit};S/B in-plane', nPtBins, ptBinsArr)
        hRawYieldsSoverBOut = TH1D('hRawYieldsSoverBOut', f';{ptTit};S/B out-of-plane', nPtBins, ptBinsArr)
        hSigmaSecPeakFitIn = TH1D('hSigmaSecPeakFitIn', f';{ptTit};width second peak in-plane', nPtBins, ptBinsArr)
        hSigmaSecPeakFitOut = TH1D('hSigmaSecPeakFitOut', f';{ptTit};width second peak out-of-plane',
                                   nPtBins, ptBinsArr)
        hvnSimFit = TH1D('hvnSimFit',f';{ptTit};V2 (Delta Phi)', nPtBins, ptBinsArr)

        SetObjectStyle(hRawYieldsIn, color=kRed, markerstyle=kFullCircle)
        SetObjectStyle(hRawYieldsOut, color=kAzure, markerstyle=kOpenCircle)
        SetObjectStyle(hSigmaIn, color=kRed, markerstyle=kFullCircle)
        SetObjectStyle(hSigmaOut, color=kAzure, markerstyle=kOpenCircle)
        SetObjectStyle(hMeanIn, color=kRed, markerstyle=kFullCircle)
        SetObjectStyle(hMeanOut, color=kAzure, markerstyle=kOpenCircle)
        SetObjectStyle(hRedChi2In, color=kRed, markerstyle=kFullCircle)
        SetObjectStyle(hRedChi2Out, color=kAzure, markerstyle=kOpenCircle)
        SetObjectStyle(hProbIn, color=kRed, markerstyle=kFullCircle)
        SetObjectStyle(hProbOut, color=kAzure, markerstyle=kOpenCircle)
        SetObjectStyle(hRawYieldsSignificanceIn, color=kRed, markerstyle=kFullCircle)
        SetObjectStyle(hRawYieldsSignificanceOut, color=kAzure, markerstyle=kOpenCircle)
        SetObjectStyle(hRawYieldsSoverBIn, color=kRed, markerstyle=kFullCircle)
        SetObjectStyle(hRawYieldsSoverBOut, color=kAzure, markerstyle=kOpenCircle)
        SetObjectStyle(hSigmaSecPeakFitIn, color=kRed, markerstyle=kFullCircle)
        SetObjectStyle(hSigmaSecPeakFitOut, color=kAzure, markerstyle=kOpenCircle)
        SetObjectStyle(hvnSimFit, color=kBlack, markerstyle=kFullCircle)

    gvnSimFit = TGraphAsymmErrors(1)
    gvnSimFit.SetName('gvnSimFit')
    gvnSimFitSecPeak = TGraphAsymmErrors(1)
    gvnSimFitSecPeak.SetName('gvnSimFitSecPeak')
    gvnUnc = TGraphAsymmErrors(1)
    gvnUnc.SetName('gvnUnc')
    gvnUncSecPeak = TGraphAsymmErrors(1)
    gvnUncSecPeak.SetName('gvnUncSecPeak')
    SetObjectStyle(gvnSimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(gvnSimFitSecPeak, color=kRed, markerstyle=kOpenCircle)
    SetObjectStyle(gvnUnc, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(gvnUncSecPeak, color=kBlack, markerstyle=kOpenCircle)

    # create canvases
    canvSizes = [1920, 1080]
    nMaxCanvases = 10 # do not put more than 20 bins per canvas to make them visible
    nCanvases = 1
    if nPtBins == 1:
        canvSizes = [500, 500]
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)

    if vn_method == 'sp' or vn_method == 'ep':
        cSimFit = []
        for i in range(nPtBins):
            ptLow = ptMins[i]
            ptHigh = ptMaxs[i]
            cSimFit.append(TCanvas(f'cSimFit_Pt{ptLow}_{ptHigh}', f'cSimFit_Pt{ptLow}_{ptHigh}', 400, 900))
            cSimFit[-1].Divide(1, 2)
    else:
        cMass, cResiduals = [], []
        cMass = TCanvas('cMass', 'cMass', canvSizes[0], canvSizes[1])
        nPads = nPtBins if nCanvases == 1 else nMaxCanvases
        DivideCanvas(cMass, nPads)
    canvVn = TCanvas('cVn', 'cVn', 900, 900)
    canvVnUnc = TCanvas('canvVnUnc', 'canvVnUnc', 900, 900)

    #_____________________________________________________
    # Vn estimation with Scalar Product / Event Plane
    if vn_method == 'sp' or vn_method == 'ep':
        massFitter, vnFitter = [], []
        for iPt, (hM, hV, ptMin, ptMax, reb, sgnEnum, bkgEnum, bkgVnEnum, secPeak, massMin, massMax) in enumerate(
                zip(hMass, hVn, ptMins, ptMaxs, rebins, SgnFunc, BkgFunc, BkgFuncVn, inclSecPeak, massMins, massMaxs)):
            iCanv = iPt
            hMassForFit.append(TH1F())
            hVnForFit.append(TH1F())
            RebinHisto(hM, reb).Copy(hMassForFit[iPt]) #to cast TH1D to TH1F
            hMassForFit[iPt].SetDirectory(0)
            xbins = np.asarray(hV.GetXaxis().GetXbins())
            hDummy = TH1F('hDummy', '', len(xbins)-1, xbins)
            for iBin in range(1, hV.GetNbinsX()+1):
                hDummy.SetBinContent(iBin, hV.GetBinContent(iBin))
                hDummy.SetBinError(iBin, hV.GetBinError(iBin))
            hVnForFit[iPt] = hDummy
            hVnForFit[iPt].SetDirectory(0)
            hVnForFit[iPt].GetXaxis().SetTitle(massAxisTit)
            hVnForFit[iPt].GetYaxis().SetTitle(f'#it{{v}}{harmonic}')
            binWidth = hMassForFit[iPt].GetBinWidth(1)
            hMassForFit[iPt].SetTitle((f'{ptMin:0.1f} < #it{{p}}_{{T}} < {ptMax:0.1f} GeV/#it{{c}};{massAxisTit};'
                                       f'Counts per {binWidth*1000:.0f} MeV/#it{{c}}^{{2}}'))
            hMassForFit[iPt].SetName(f'MassForFit{iPt}')
            SetObjectStyle(hMassForFit[iPt], color=kBlack, markerstyle=kFullCircle, markersize=1)
            SetObjectStyle(hVnForFit[iPt], color=kBlack, markerstyle=kFullCircle, markersize=0.8)

            print(f'Fitting {ptMin} - {ptMax} GeV/c')
            vnFitter.append(VnVsMassFitter(hMassForFit[iPt], hVnForFit[iPt],
                                                massMin, massMax, bkgEnum, sgnEnum, bkgVnEnum))
            vnFitter[iPt].SetHarmonic(harmonic)
            #hMassForFit[iPt].DrawCopy()
            #input()

            #_____________________________________________________
            # set the parameters for the fit
            # Mean
            vnFitter[iPt].SetInitialGaussianMean(massForFit, 1)
            if fixMean[iPt]:
                vnFitter[iPt].FixMeanFromMassFit()
            # Sigma
            vnFitter[iPt].SetInitialGaussianSigma(fitConfig['Sigma'][iPt], 1)
            if fixSigma[iPt]:
                vnFitter[iPt].FixSigmaFromMassFit()
            # nSigma4SB
            if 'NSigma4SB' in fitConfig:
                vnFitter[iPt].SetNSigmaForVnSB(fitConfig['NSigma4SB'][iPt])
                print(f'NSigma4SB = {fitConfig["NSigma4SB"][iPt]}')
            # Second peak (Ds specific)
            if secPeak and particleName == 'Ds':
                vnFitter[iPt].IncludeSecondGausPeak(massDplus, False, fitConfig['SigmaSecPeak'][iPt], False, 1)
                if fixSigma[iPt]:
                    vnFitter[iPt].FixSigma2GausFromMassFit()
            vnFitter[iPt].FixFrac2GausFromMassFit()
            # TODO: Add reflections for D0
            # Reflections for D0
            if useRefl:
                SoverR = (hMCRefl[iPt].Integral(hMCRefl[iPt].FindBin(massMin*1.0001),hMCRefl[iPt].FindBin(massMax*0.9999)))/(
                    hMCSgn[iPt].Integral(hMCSgn[iPt].FindBin(massMin*1.0001),hMCSgn[iPt].FindBin(massMax*0.9999)))
                vnFitter[iPt].SetTemplateReflections(hMCRefl[iPt],reflFuncStr,massMin,massMax)
                vnFitter[iPt].SetFixReflOverS(SoverR)
                vnFitter[iPt].SetReflVnOption(0) # kSameVnSignal
            if fitConfig.get('IncludeKDETempls'):
                vnFitter[iPt].SetKDETemplates(KDEtemplatesFuncts[iPt], fitConfig['TemplsTreeNames'],
                                              fitConfig['InitWeights'][iPt], fitConfig['MinWeights'][iPt], 
                                              fitConfig['MaxWeights'][iPt])

            # collect fit results
            vnFitter[iPt].SimultaneousFit(False)
            vnResults = get_vnfitter_results(vnFitter[iPt], secPeak, useRefl)
            fTotFuncMass.append(vnResults['fTotFuncMass'])
            fTotFuncVn.append(vnResults['fTotFuncVn'])
            fSgnFuncMass.append(vnResults['fSgnFuncMass'])
            fBkgFuncMass.append(vnResults['fBkgFuncMass'])
            fBkgFuncVn.append(vnResults['fBkgFuncVn'])
            if fitConfig.get('IncludeKDETempls'):
                fMassTemplFuncts[iPt] = vnResults['fMassTemplFuncts']
                fVnTemplFuncts[iPt] = vnResults['fVnTemplFuncts']

            if useRefl:
                fMassBkgRflFunc.append(vnResults['fMassBkgRflFunc'])
                hRel.append(vnResults['fMassRflFunc'])

            hSigmaSimFit.SetBinContent(iPt+1, vnResults['sigma'])
            hSigmaSimFit.SetBinError(iPt+1, vnResults['sigmaUnc'])
            hMeanSimFit.SetBinContent(iPt+1, vnResults['mean'])
            hMeanSimFit.SetBinError(iPt+1, vnResults['meanUnc'])
            hRedChi2SimFit.SetBinContent(iPt+1, vnResults['chi2'])
            hRedChi2SimFit.SetBinError(iPt+1, 1.e-20)
            hProbSimFit.SetBinContent(iPt+1, vnResults['prob'])
            hProbSimFit.SetBinError(iPt+1, 1.e-20)
            hRawYieldsSimFit.SetBinContent(iPt+1, vnResults['ry'])
            hRawYieldsSimFit.SetBinError(iPt+1, vnResults['ryUnc'])
            hRawYieldsTrueSimFit.SetBinContent(iPt+1, vnResults['ryTrue'])
            hRawYieldsTrueSimFit.SetBinError(iPt+1, vnResults['ryTrueUnc'])
            hRawYieldsSignificanceSimFit.SetBinContent(iPt+1, vnResults['signif'])
            hRawYieldsSignificanceSimFit.SetBinError(iPt+1, vnResults['signifUnc'])
            hvnSimFit.SetBinContent(iPt+1, vnResults['vn'])
            hvnSimFit.SetBinError(iPt+1, vnResults['vnUnc'])
            gvnSimFit.SetPoint(iPt, (ptMin+ptMax)/2, vnResults['vn'])
            gvnSimFit.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, vnResults['vnUnc'], vnResults['vnUnc'])
            gvnUnc.SetPoint(iPt, (ptMin+ptMax)/2, vnResults['vnUnc'])
            gvnUnc.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, 1.e-20, 1.e-20)

            if secPeak:
                hMeanSecPeakFitMass.SetBinContent(iPt+1, vnResults['secPeakMeanMass'])
                hMeanSecPeakFitMass.SetBinError(iPt+1, vnResults['secPeakMeanMassUnc'])
                hSigmaSecPeakFitMass.SetBinContent(iPt+1, vnResults['secPeakSigmaMass'])
                hSigmaSecPeakFitMass.SetBinError(iPt+1, vnResults['secPeakSigmaMassUnc'])
                hMeanSecPeakFitVn.SetBinContent(iPt+1, vnResults['secPeakMeanVn'])
                hMeanSecPeakFitVn.SetBinError(iPt+1, vnResults['secPeakMeanVnUnc'])
                hSigmaSecPeakFitVn.SetBinContent(iPt+1, vnResults['secPeakSigmaVn'])
                hSigmaSecPeakFitVn.SetBinError(iPt+1, vnResults['secPeakSigmaVnUnc'])
                gvnSimFitSecPeak.SetPoint(iPt, (ptMin+ptMax)/2, vnResults['vnSecPeak'])
                gvnSimFitSecPeak.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2,
                                               vnResults['vnSecPeakUnc'],
                                               vnResults['vnSecPeakUnc'])
                gvnUncSecPeak.SetPoint(iPt, (ptMin+ptMax)/2, vnResults['vnSecPeakUnc'])
                gvnUncSecPeak.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, 1.e-20, 1.e-20)

            if vnResults['vn'] != 0:
                cSimFit[iPt].cd(1)
                hMassForFit[iPt].GetYaxis().SetRangeUser(0.2*hMassForFit[iPt].GetMinimum(),
                                                         1.6*hMassForFit[iPt].GetMaximum())
                hMassForFit[iPt].GetYaxis().SetMaxDigits(3)
                hMassForFit[iPt].GetXaxis().SetRangeUser(massMin, massMax)
                hMassForFit[iPt].Draw('E')
                SetObjectStyle(fBkgFuncMass[iPt], color=kOrange+1, linestyle=9, linewidth=2)
                SetObjectStyle(fTotFuncMass[iPt], color=kAzure+4, linewidth=3)
                SetObjectStyle(fSgnFuncMass[iPt], fillcolor=kAzure+4, fillstyle=3245, linewidth=0)
                if useRefl:
                    SetObjectStyle(hRel[iPt], fillcolor=kGreen+1, fillstyle=3254, linewidth=0)
                    SetObjectStyle(fMassBkgRflFunc[iPt], color=kRed+1, linestyle=7, linewidth=2)
                fSgnFuncMass[iPt].Draw('fc same')
                fBkgFuncMass[iPt].Draw('same')
                fTotFuncMass[iPt].Draw('same')
                if useRefl:
                    fMassBkgRflFunc[iPt].Draw('same')
                    hRel[iPt].Draw('same')
                latex.DrawLatex(0.18, 0.80, f'#mu = {vnResults["mean"]:.3f} #pm {vnResults["meanUnc"]:.3f} GeV/c^{2}')
                latex.DrawLatex(0.18, 0.75, f'#sigma = {vnResults["sigma"]:.3f} #pm {vnResults["sigmaUnc"]:.3f} GeV/c^{2}')
                latex.DrawLatex(0.18, 0.70, f'S = {vnResults["ry"]:.0f} #pm {vnResults["ryUnc"]:.0f}')
                latex.DrawLatex(0.18, 0.65, f'S/B (3#sigma) = {vnResults["ry"]/vnResults["bkg"]:.2f}')
                latex.DrawLatex(0.18, 0.60, f'Signif. (3#sigma) = {round(vnResults["signif"], 2)}')
                if useRefl:
                    latex.DrawLatex(0.18, 0.20, f'RoverS = {SoverR:.2f}')
                if fitConfig.get('IncludeKDETempls'):
                    for iMassTemplFunct, massTemplFunct in enumerate(fMassTemplFuncts[iPt]):
                        SetObjectStyle(massTemplFunct, color=kMagenta+iMassTemplFunct*2, linewidth=3)
                        massTemplFunct.SetLineColor(1)
                        massTemplFunct.Draw('same')
                        cSimFit[iCanv].Modified()
                        cSimFit[iCanv].Update()
                cSimFit[iPt].cd(2)
                hVnForFit[iPt].GetYaxis().SetRangeUser(-1, 1)
                hVnForFit[iPt].GetYaxis().SetTitle(f'#it{{v}}_{{{harmonic}}} ({vn_method})')
                hVnForFit[iPt].GetXaxis().SetRangeUser(massMin, massMax)
                hVnForFit[iPt].Draw('E')
                SetObjectStyle(fBkgFuncVn[iPt], color=kOrange+1, linestyle=9, linewidth=2)
                SetObjectStyle(fTotFuncVn[iPt], color=kAzure+4, linewidth=3)
                fBkgFuncVn[iPt].Draw('same')
                fTotFuncVn[iPt].Draw('same')
                latex.DrawLatex(0.18, 0.23, f'#it{{R}}_{{{harmonic}}} = {reso:.3f}')
                latex.DrawLatex(0.18, 0.18, f'#chi^{{2}}/ndf = {vnResults["chi2"]:.2f}')
                latex.DrawLatex(0.18, 0.80,
                                f'#it{{v}}{harmonic}({particleName}) = {vnResults["vn"]:.3f} #pm {vnResults["vnUnc"]:.3f}')
                if secPeak:
                    latex.DrawLatex(0.18, 0.75,
                                    f'#it{{v}}{harmonic}(D^{{+}}) = {vnResults["vnSecPeak"]:.3f} #pm {vnResults["vnSecPeakUnc"]:.3f}')
                if fitConfig.get('IncludeKDETempls'):
                    if fitConfig.get('drawvncomps'):
                        for iVnTemplFunct, vnTemplFunct in enumerate(fVnTemplFuncts[iPt]):
                            vnTemplFunct.SetLineColor(iVnTemplFunct+1)
                            vnTemplFunct.Draw('same')
                            cSimFit[iCanv].Modified()
                            cSimFit[iCanv].Update()
                cSimFit[iCanv].Modified()
                cSimFit[iCanv].Update()

    #_____________________________________________________
    # Mass fit
    else:
        massFitterIns, massFitterOuts = [], []
        for iPt, (hMassIn, hMassOut,\
                  ptMin, ptMax,\
                  reb, sgnEnum, bkgEnum, bkgVnEnum,\
                  secPeak, massMin, massMax) in enumerate(zip(hMassIns, hMassOuts,
                                                              ptMins, ptMaxs,
                                                              rebins, SgnFunc, BkgFunc,
                                                              BkgFuncVn, inclSecPeak,
                                                              massMins, massMaxs)):
            iCanv = iPt
            hMassInsForFit.append(TH1F())
            hMassOutsForFit.append(TH1F())
            RebinHisto(hMassIn, reb).Copy(hMassInsForFit[iPt])
            RebinHisto(hMassOut, reb).Copy(hMassOutsForFit[iPt])
            hMassInsForFit[iPt].SetDirectory(0)
            hMassOutsForFit[iPt].SetDirectory(0)
            binWidth = hMassInsForFit[iPt].GetBinWidth(1)
            hMassInsForFit[iPt].SetTitle((f'{ptMin:0.1f} < #it{{p}}_{{T}} < {ptMax:0.1f} GeV/#it{{c}};{massAxisTit};'
                                            f'Counts per {binWidth*1000:.0f} MeV/#it{{c}}^{{2}}'))
            hMassInsForFit[iPt].SetName(f'MassInForFit{iPt}')
            hMassOutsForFit[iPt].SetTitle((f'{ptMin:0.1f} < #it{{p}}_{{T}} < {ptMax:0.1f} GeV/#it{{c}};{massAxisTit};'
                                            f'Counts per {binWidth*1000:.0f} MeV/#it{{c}}^{{2}}'))
            hMassOutsForFit[iPt].SetName(f'MassOutForFit{iPt}')
            SetObjectStyle(hMassInsForFit[iPt], color=kRed-3, markerstyle=kFullCircle, markersize=0.8)
            SetObjectStyle(hMassOutsForFit[iPt], color=kAzure-3, markerstyle=kOpenCircle, markersize=0.8)

            print(f'Fitting {ptMin} - {ptMax} GeV/c')
            massFitterIns.append(InvMassFitter(hMassInsForFit[iPt],  massMin, massMax, bkgEnum, sgnEnum))
            massFitterOuts.append(InvMassFitter(hMassOutsForFit[iPt],  massMin, massMax, bkgEnum, sgnEnum))
            if degPol[iPt] > 0:
                massFitterIns[iPt].SetPolDegreeForBackgroundFit(degPol[iPt])
                massFitterOuts[iPt].SetPolDegreeForBackgroundFit(degPol[iPt])
            massFitterIns[iPt].SetUseLikelihoodFit()
            massFitterOuts[iPt].SetUseLikelihoodFit()
            if fitConfig['BoundMean']:
                massFitterIns[iPt].SetBoundGaussianMean(massForFit, massMin, massMax)
                massFitterOuts[iPt].SetBoundGaussianMean(massForFit, massMin, massMax)
            else:
                massFitterIns[iPt].SetInitialGaussianMean(massForFit)
                massFitterOuts[iPt].SetInitialGaussianMean(massForFit)
            if fitConfig['FixSigmaRatio']:
                massFitterIns[iPt].SetFixRatio2GausSigma(
                    hSigmaToFix.GetBinContent(iPt+1)/hSigmaToFix2.GetBinContent(iPt+1))
                massFitterOuts[iPt].SetFixRatio2GausSigma(
                    hSigmaToFix.GetBinContent(iPt+1)/hSigmaToFix2.GetBinContent(iPt+1))

            if fixSigma[iPt]:
                if isinstance(fitConfig['SigmaMultFactor'], (float, int)):
                    massFitterIns[iPt].SetFixGaussianSigma(
                        hSigmaToFix.GetBinContent(iPt+1)*fitConfig['SigmaMultFactor'])
                    massFitterOuts[iPt].SetFixGaussianSigma(
                        hSigmaToFix.GetBinContent(iPt+1)*fitConfig['SigmaMultFactor'])
                else:
                    if fitConfig['SigmaMultFactor'] == 'MinusUnc':
                        massFitterIns[iPt].SetFixGaussianSigma(
                            hSigmaToFix.GetBinContent(iPt+1)-hSigmaToFix.GetBinError(iPt+1))
                        massFitterOuts[iPt].SetFixGaussianSigma(
                            hSigmaToFix.GetBinContent(iPt+1)-hSigmaToFix.GetBinError(iPt+1))
                    elif fitConfig['SigmaMultFactor'] == 'PlusUnc':
                        massFitterIns[iPt].SetFixGaussianSigma(
                            hSigmaToFix.GetBinContent(iPt+1)+hSigmaToFix.GetBinError(iPt+1))
                        massFitterOuts[iPt].SetFixGaussianSigma(
                            hSigmaToFix.GetBinContent(iPt+1)+hSigmaToFix.GetBinError(iPt+1))
                    else:
                        print('WARNING: impossible to fix sigma! Wrong mult factor set in config file!')
            else:
                if hSigmaToFix:
                    massFitterIns[iPt].SetInitialGaussianSigma(
                        hSigmaToFix.GetBinContent(iPt+1)*fitConfig['SigmaMultFactor'])
                    massFitterOuts[iPt].SetInitialGaussianSigma(
                        hSigmaToFix.GetBinContent(iPt+1)*fitConfig['SigmaMultFactor'])
                else:
                    if particleName == 'Dstar':
                        massFitterIns[iPt].SetInitialGaussianSigma(0.001)
                        massFitterOuts[iPt].SetInitialGaussianSigma(0.001)
                    else:
                        massFitterIns[iPt].SetInitialGaussianSigma(0.008)
                        massFitterOuts[iPt].SetInitialGaussianSigma(0.008)

            if secPeak and particleName == 'Ds':
                massFitterIns[iPt].IncludeSecondGausPeak(massDplus, False, fitConfig['SigmaSecPeak'][iPt], True)
                massFitterOuts[iPt].IncludeSecondGausPeak(massDplus, False, fitConfig['SigmaSecPeak'][iPt], True)
            # TODO: Add reflections for D0
            # Reflections for D0
            if useRefl:
                SoverR = (hMCRefl[iPt].Integral(hMCRefl[iPt].FindBin(massMin*1.0001),hMCRefl[iPt].FindBin(massMax*0.9999)))/(
                    hMCSgn[iPt].Integral(hMCSgn[iPt].FindBin(massMin*1.0001),hMCSgn[iPt].FindBin(massMax*0.9999)))
                massFitterIns[iPt].SetTemplateReflections(hMCRefl[iPt],reflFuncStr,massMin,massMax)
                massFitterIns[iPt].SetFixReflOverS(SoverR)
                massFitterOuts[iPt].SetTemplateReflections(hMCRefl[iPt],reflFuncStr,massMin,massMax)
                massFitterOuts[iPt].SetFixReflOverS(SoverR)
            massFitterIns[iPt].MassFitter(False)
            massFitterOuts[iPt].MassFitter(False)

            # collect fit results
            rawyield_in = massFitterIns[iPt].GetRawYield()
            rawyielderr_in = massFitterIns[iPt].GetRawYieldError()
            sigma_in = massFitterIns[iPt].GetSigma()
            sigmaerr_in = massFitterIns[iPt].GetSigmaUncertainty()
            mean_in = massFitterIns[iPt].GetMean()
            meanerr_in = massFitterIns[iPt].GetMeanUncertainty()
            redchi2_in = massFitterIns[iPt].GetReducedChiSquare()
            signif_in, signiferr_in = ctypes.c_double(), ctypes.c_double()
            sgn_in, sgnerr_in = ctypes.c_double(), ctypes.c_double()
            bkg_in, bkgerr_in = ctypes.c_double(), ctypes.c_double()
            massFitterIns[iPt].Significance(3, signif_in, signiferr_in)
            massFitterIns[iPt].Signal(3, sgn_in, sgnerr_in)
            massFitterIns[iPt].Background(3, bkg_in, bkgerr_in)

            rawyield_out = massFitterOuts[iPt].GetRawYield()
            rawyielderr_out = massFitterOuts[iPt].GetRawYieldError()
            sigma_out = massFitterOuts[iPt].GetSigma()
            sigmaerr_out = massFitterOuts[iPt].GetSigmaUncertainty()
            mean_out = massFitterOuts[iPt].GetMean()
            meanerr_out = massFitterOuts[iPt].GetMeanUncertainty()
            redchi2_out = massFitterOuts[iPt].GetReducedChiSquare()
            signif_out, signiferr_out = ctypes.c_double(), ctypes.c_double()
            sgn_out, sgnerr_out = ctypes.c_double(), ctypes.c_double()
            bkg_out, bkgerr_out = ctypes.c_double(), ctypes.c_double()
            massFitterOuts[iPt].Significance(3, signif_out, signiferr_out)
            massFitterOuts[iPt].Signal(3, sgn_out, sgnerr_out)
            massFitterOuts[iPt].Background(3, bkg_out, bkgerr_out)

            hRawYieldsIn.SetBinContent(iPt+1, rawyield_in)
            hRawYieldsIn.SetBinError(iPt+1, rawyielderr_in)
            hSigmaIn.SetBinContent(iPt+1, sigma_in)
            hSigmaIn.SetBinError(iPt+1, sigmaerr_in)
            hMeanIn.SetBinContent(iPt+1, mean_in)
            hMeanIn.SetBinError(iPt+1, meanerr_in)
            hRedChi2In.SetBinContent(iPt+1, redchi2_in)
            hRedChi2In.SetBinError(iPt+1, 1.e-20)
            hRawYieldsSignificanceIn.SetBinContent(iPt+1, signif_in.value)
            hRawYieldsSignificanceIn.SetBinError(iPt+1, signiferr_in.value)
            hRawYieldsSoverBIn.SetBinContent(iPt+1, sgn_in.value/bkg_in.value)
            hRawYieldsSoverBIn.SetBinError(iPt+1, 1.e-20)
            hRawYieldsOut.SetBinContent(iPt+1, rawyield_out)
            hRawYieldsOut.SetBinError(iPt+1, rawyielderr_out)
            hSigmaOut.SetBinContent(iPt+1, sigma_out)
            hSigmaOut.SetBinError(iPt+1, sigmaerr_out)
            hMeanOut.SetBinContent(iPt+1, mean_out)
            hMeanOut.SetBinError(iPt+1, meanerr_out)
            hRedChi2Out.SetBinContent(iPt+1, redchi2_out)
            hRedChi2Out.SetBinError(iPt+1, 1.e-20)
            hRawYieldsSignificanceOut.SetBinContent(iPt+1, signif_out.value)
            hRawYieldsSignificanceOut.SetBinError(iPt+1, signiferr_out.value)
            hRawYieldsSoverBOut.SetBinContent(iPt+1, sgn_out.value/bkg_out.value)
            hRawYieldsSoverBOut.SetBinError(iPt+1, 1.e-20)

            # plot the results
            cMass.cd(iPt+1)
            hMassInsForFit[iPt].GetYaxis().SetRangeUser(0, hMassInsForFit[iPt].GetMaximum()*2)
            hMassInsForFit[iPt].GetXaxis().SetRangeUser(massMin, massMax)
            hMassInsForFit[iPt].Draw('PE')
            hMassOutsForFit[iPt].Draw('PE same')
            fTotFuncMassIn = massFitterIns[iPt].GetMassFunc()
            fTotFuncMassOut = massFitterOuts[iPt].GetMassFunc()
            fBkgFuncMassIn = massFitterIns[iPt].GetBackgroundRecalcFunc()
            fBkgFuncMassOut = massFitterOuts[iPt].GetBackgroundRecalcFunc()
            SetObjectStyle(fTotFuncMassIn, color=kRed, linestyle=2, linewidth=3)
            SetObjectStyle(fTotFuncMassOut, color=kAzure, linestyle=9, linewidth=3)
            SetObjectStyle(fBkgFuncMassIn, color=kGray+1, linestyle=9, linewidth=1)
            SetObjectStyle(fBkgFuncMassOut, color=kGray+1, linestyle=9, linewidth=1)
            fBkgFuncMassIn.Draw('same')
            fBkgFuncMassOut.Draw('same')
            fTotFuncMassIn.Draw('same')
            fTotFuncMassOut.Draw('same')
            if iPt == 0:
                leg = TLegend(0.15, 0.3, 0.5, 0.5)
                leg.SetBorderSize(0)
                leg.SetFillStyle(0)
                leg.AddEntry(hMassInsForFit[iPt], 'In-plane', 'p')
                leg.AddEntry(hMassOutsForFit[iPt], 'Out-of-plane', 'p')
                leg.Draw()
            latex.SetTextColor(kRed)
            latex.DrawLatex(0.15, 0.80, f'#mu = {mean_in:.3f} #pm {meanerr_in:.3f} GeV/c^{2}')
            latex.DrawLatex(0.15, 0.75, f'#sigma = {sigma_in:.3f} #pm {sigmaerr_in:.3f} GeV/c^{2}')
            latex.DrawLatex(0.15, 0.70, f'S = {rawyield_in:.0f} #pm {rawyielderr_in:.0f}')
            latex.DrawLatex(0.15, 0.65, f'S/B (3#sigma) = {rawyield_in/bkg_in.value:.2f}')
            latex.DrawLatex(0.15, 0.60, f'#chi^{{2}}/ndf = {redchi2_in:.2f}')
            latex.DrawLatex(0.15, 0.55, f'Signif. (3#sigma) = {signif_in.value:.2f} #pm {signiferr_in.value:.2f}')
            latex.SetTextColor(kAzure)
            latex.DrawLatex(0.6, 0.80, f'#mu = {mean_out:.3f} #pm {meanerr_out:.3f} GeV/c^{2}')
            latex.DrawLatex(0.6, 0.75, f'#sigma = {sigma_out:.3f} #pm {sigmaerr_out:.3f} GeV/c^{2}')
            latex.DrawLatex(0.6, 0.70, f'S = {rawyield_out:.0f} #pm {rawyielderr_out:.0f}')
            latex.DrawLatex(0.6, 0.65, f'S/B = {rawyield_out/bkg_out.value:.2f}')
            latex.DrawLatex(0.6, 0.60, f'#chi^{{2}}/ndf = {redchi2_out:.2f}')
            latex.DrawLatex(0.6, 0.55, f'Signif. (3#sigma) = {signif_out.value:.2f} #pm {signiferr_out.value:.2f}')
            cMass.Modified()
            cMass.Update()

            # vn
            vn, vnUnc = get_ep_vn(harmonic,
                                  rawyield_in, rawyielderr_in,
                                  rawyield_out, rawyielderr_out,
                                  reso)
            hvnSimFit.SetBinContent(iPt+1, vn)
            hvnSimFit.SetBinError(iPt+1, vnUnc)
            gvnSimFit.SetPoint(iPt, (ptMin+ptMax)/2, vn)
            gvnSimFit.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, vnUnc, vnUnc)
            gvnUnc.SetPoint(iPt, (ptMin+ptMax)/2, vnUnc)
            gvnUnc.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, 1.e-20, 1.e-20)
    
    canvVn.cd().SetLogx()
    hframe = canvVn.DrawFrame(0.5, -0.5, gvnSimFit.GetXaxis().GetXmax()+0.5, 0.5,
                              f';#it{{p}}_{{T}} (GeV/c); v_{{{harmonic}}} ({vn_method})')
    hframe.GetYaxis().SetDecimals()
    hframe.GetXaxis().SetNdivisions(504)
    hframe.GetXaxis().SetMoreLogLabels()
    gPad.SetGridy()
    gvnSimFit.Draw('same pez')
    if secPeak:
        gvnSimFitSecPeak.Draw('pez same')
    latex.DrawLatex(0.2, 0.2, f'#it{{R}}_{{{harmonic}}} = {reso:.3f}')
    latex.DrawLatexNDC(0.20, 0.80, 'This work')
    latex.DrawLatexNDC(0.20, 0.75, f'Pb#minusPb #sqrt{{#it{{s}}_{{NN}}}} = 5.36 TeV ({centMinMax[0]}#minus{centMinMax[1]}%)')
    latex.DrawLatexNDC(0.20, 0.70, decay)
    canvVn.Modified()
    canvVn.Update()
    canvVnUnc.cd()
    gvnUnc.Draw('apez same')
    if secPeak:
        gvnUncSecPeak.Draw('pez same')
    canvVnUnc.Modified()
    canvVnUnc.Update()
    if not batch:
        input('Press Enter to continue...')

    #save output histos
    print(f'Saving output to {outputdir}')
    if vn_method == 'sp' or vn_method == 'ep':
        for iPt, (ptMin, ptMax) in enumerate(zip(ptMins, ptMaxs)):
            if iPt == 0:
                suffix_pdf = '('
            elif iPt == nPtBins-1:
                suffix_pdf = ')'
            else:
                suffix_pdf = ''
            if len(ptMins)==1:
                cSimFit[iPt].SaveAs(f'{outputdir}/SimFit{suffix}_{particleName}.pdf')
            else:
                cSimFit[iPt].SaveAs(f'{outputdir}/SimFit{suffix}_{particleName}.pdf{suffix_pdf}')
    outfile_name = f'{outputdir}/raw_yields{suffix}.root'
    outFile = TFile(outfile_name, 'recreate')
    if vn_method == 'sp' or vn_method == 'ep':
        for canv in cSimFit:
            canv.Write()
        for hist in hMass:
            hist.Write()
        for hist in hVn:
            hist.Write()
        for ipt, (ptmin, ptmax) in enumerate(zip(ptMins, ptMaxs)):
            fTotFuncMass[ipt].Write(f'fTotFuncMass_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fTotFuncVn[ipt].Write(f'fTotFuncVn_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fSgnFuncMass[ipt].Write(f'fSgnFuncMass_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fBkgFuncMass[ipt].Write(f'fBkgFuncMass_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fBkgFuncVn[ipt].Write(f'fBkgFuncVn_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            if fitConfig.get('IncludeKDETempls'):
                for iFlag in range(len(KDEtemplatesFuncts[ipt])):
                    KDEtemplatesFuncts[ipt][iFlag].Write(f'{fitConfig["TemplsTreeNames"][iFlag]}_pt{ptmin*10:.0f}_{ptmax*10:.0f}_flag{fitConfig["TemplsFlags"][iFlag]}')
                
        hSigmaSimFit.Write()
        hMeanSimFit.Write()
        hMeanSecPeakFitMass.Write()
        hMeanSecPeakFitVn.Write()
        hSigmaSecPeakFitMass.Write()
        hSigmaSecPeakFitVn.Write()
        hRawYieldsSimFit.Write()
        hRawYieldsTrueSimFit.Write()
        hRawYieldsSecPeakSimFit.Write()
        hRawYieldsSignificanceSimFit.Write()
        hRawYieldsSoverBSimFit.Write()
        hRedChi2SimFit.Write()
        hProbSimFit.Write()
        hRedChi2SBVnPrefit.Write()
        hProbSBVnPrefit.Write()
        hvnSimFit.Write()
    else:
        cMass.SaveAs(f'{outputdir}/MassFit{suffix}_{particleName}.pdf')
        cMass.Write()
        for hist in hMassInsForFit:
            hist.Write()
        for hist in hMassOutsForFit:
            hist.Write()
        for ipt, (ptmin, ptmax) in enumerate(zip(ptMins, ptMaxs)):
            print(f'Writing {ptmin} - {ptmax} GeV/c')
            fTotFuncMassIn = massFitterIns[ipt].GetMassFunc()
            fTotFuncMassOut = massFitterOuts[ipt].GetMassFunc()
            fTotFuncMassIn.Write(f'fTotFuncMassIn_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fTotFuncMassOut.Write(f'fTotFuncMassOut_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
        hRawYieldsIn.Write()
        hRawYieldsOut.Write()
        hSigmaIn.Write()
        hSigmaOut.Write()
        hMeanIn.Write()
        hMeanOut.Write()
        hRedChi2In.Write()
        hRedChi2Out.Write()
        hRawYieldsSignificanceIn.Write()
        hRawYieldsSignificanceOut.Write()
        hRawYieldsSoverBIn.Write()
        hRawYieldsSoverBOut.Write()
        hSigmaSecPeakFitIn.Write()
        hSigmaSecPeakFitOut.Write()
        hvnSimFit.Write()

    gvnSimFit.Write()
    gvnUnc.Write()
    if secPeak:
        gvnSimFitSecPeak.Write()
        gvnUncSecPeak.Write()
    hist_reso.Write()

    outFile.Close()

    if not batch:
        input('Press enter to exit')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('fitConfigFileName', metavar='text', default='config_Ds_Fit.yml')
    parser.add_argument('centClass', metavar='text', default='')
    parser.add_argument('inFileName', metavar='text', default='')
    parser.add_argument("--outputdir", "-o", metavar="text",
                        default=".", help="output directory")
    parser.add_argument("--suffix", "-s", metavar="text",
                        default="", help="suffix for output files")
    parser.add_argument('--vn_method', '-vn', metavar='text', default='sp')
    parser.add_argument('--batch', help='suppress video output', action='store_true')
    args = parser.parse_args()

    get_vn_vs_mass(
        args.fitConfigFileName,
        args.centClass,
        args.inFileName,
        args.outputdir,
        args.suffix,
        args.vn_method,
        args.batch
    )