import ctypes
import ROOT
import yaml
import argparse
import sys
import os
import numpy as np
from ROOT import TFile, TCanvas, TH1F, TLegend, gROOT  # pylint: disable=import-error,no-name-in-module
from flow_analysis_utils import get_centrality_bins
### please fill your path of DmeasonAnalysis
sys.path.append('../../../')
from utils.StyleFormatter import SetGlobalStyle, SetObjectStyle
from utils.AnalysisUtils import ComputeEfficiency

SetGlobalStyle(titleoffsety=1.1, maxdigits=3, topmargin=0.1,
               bottommargin=0.4, leftmargin=0.3, rightmargin=0.15,
               labelsizey=0.04, setoptstat=0, setopttitle=0,
               setdecimals=True, titleoffsetx=0.91, titlesizex=0.05)

def check_cent_sel(charm_hadron, centmin, centmax, infile):
    if charm_hadron != 'Dzero' and charm_hadron != 'Dplus' and charm_hadron != 'Ds':
        print('\033[93mWARNING: Invalid charm hadron for centrality check.\033[0m')
        sys.exit(1)

    nprongs = 2 if charm_hadron == 'Dzero' else 3
    cent_hist = infile.Get(f'hf-candidate-creator-{nprongs}prong/hSelCollisionsCent')
    for i in range(cent_hist.GetNbinsX()):
        if not (cent_hist.GetBinContent(centmin + 1) > 0 \
            and cent_hist.GetBinContent(centmin) == 0 \
            and cent_hist.GetBinContent(centmax) > 0 \
            and cent_hist.GetBinContent(centmax + 1) == 0):
            print(f'\033[91mFATAL: Invalid centrality class: [{centmin}-{centmax}]. Exit!\033[0m')
            sys.exit(1)

def compute_eff(config_file, centclass, inputFile, outputdir, suffix):
    '''
    Compute efficiency for prompt and feed-down D mesons

    Args:
    config_file: str
        configuration file
    centclass: str
        centrality class
    outputdir: str
        output directory
    suffix: str
        suffix for output files

    Returns:
    output ROOT file with efficiency histograms
    '''

    #_____________________________________________________________________________________
    # Read configuration file
    with open(config_file, 'r', encoding='utf8') as ymlconfig:
        config = yaml.load(ymlconfig, yaml.FullLoader)

    #_____________________________________________________________________________________
    # Load input files
    eff_filename = config['eff_filename']
    charm_hadron = config['Dmeson']
    infile = ROOT.TFile.Open(eff_filename)

    #_____________________________________________________________________________________
    # Check centrality selection
    centMin, centMax = get_centrality_bins(centclass)[1]
    check_cent_sel(charm_hadron, centMin, centMax, infile)

    if charm_hadron == 'Dplus':
        gen_prompt_hist = infile.Get('hf-task-dplus/hPtGenPrompt')
        gen_fd_hist = infile.Get('hf-task-dplus/hPtGenNonPrompt')
        rec_prompt_hist = infile.Get('hf-task-dplus/hPtRecSigPrompt')
        rec_fd_hist = infile.Get('hf-task-dplus/hPtRecSigNonPrompt')
    elif charm_hadron == 'Ds':
        gen_prompt_hist = infile.Get('hf-task-ds/hPtGenPrompt')
        gen_fd_hist = infile.Get('hf-task-ds/hPtGenNonPrompt')
        rec_prompt_hist = infile.Get('hf-task-ds/hPtRecSigPrompt')
        rec_fd_hist = infile.Get('hf-task-ds/hPtRecSigNonPrompt')
    #TODO: Add Dzero and Lc
    else:
        sys.exit(f'\033[91mFATAL: Invalid charm_hadron: {charm_hadron}. Exit!\033[0m')
    infile.Close()

    #_____________________________________________________________________________________
    # Compute efficiency
    eff_prompt = gen_prompt_hist.Clone('effPrompt')
    eff_prompt.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
    eff_prompt.GetYaxis().SetTitle('Acc.#times#varepsilon') # unlike run 2, in run 3 we directly have the Eff.#times Acc. from the HF task
    eff_prompt.Reset()
    eff_prompt.SetDirectory(0)
    eff_fd = gen_fd_hist.Clone('effFD')
    eff_fd.Reset()
    eff_fd.SetDirectory(0)
    for i in range(gen_prompt_hist.GetNbinsX()):
        if gen_prompt_hist.GetBinContent(i+1) > 0:
            eff, eff_unc = ComputeEfficiency(rec_prompt_hist.GetBinContent(i+1),
                                             gen_prompt_hist.GetBinContent(i+1),
                                             rec_prompt_hist.GetBinError(i+1),
                                             gen_prompt_hist.GetBinError(i+1))
            eff_prompt.SetBinContent(i+1, eff)
            eff_prompt.SetBinError(i+1, eff_unc)
        else:
            eff_prompt.SetBinContent(i+1, 0)
            eff_prompt.SetBinError(i+1, 0)
    for i in range(gen_fd_hist.GetNbinsX()):
        if gen_fd_hist.GetBinContent(i+1) > 0:
            eff, eff_unc = ComputeEfficiency(rec_fd_hist.GetBinContent(i+1),
                                             gen_fd_hist.GetBinContent(i+1),
                                             rec_fd_hist.GetBinError(i+1),
                                             gen_fd_hist.GetBinError(i+1))
            eff_fd.SetBinContent(i+1, eff)
            eff_fd.SetBinError(i+1, eff_unc)
        else:
            eff_fd.SetBinContent(i+1, 0)
            eff_fd.SetBinError(i+1, 0)

    #_____________________________________________________________________________________
    # Draw histograms
    c = ROOT.TCanvas('c', 'c', 600, 600)
    c.cd().SetLogy()
    SetObjectStyle(eff_prompt, markerstyle=20,
                   markercolor=ROOT.kOrange+1,
                   markersize=1., linecolor=ROOT.kOrange+1)
    SetObjectStyle(eff_fd, markerstyle=21,
                   linestyle=2,
                   markercolor=ROOT.kAzure+2, markersize=1.,
                   linecolor=ROOT.kAzure+2)
    eff_prompt.GetYaxis().SetRangeUser(1.e-4, 1.1)
    eff_prompt.Draw('same')
    eff_fd.Draw('same')
    leg = ROOT.TLegend(0.15, 0.7, 0.4, 0.85)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.AddEntry(eff_prompt, 'Prompt', 'lep')
    leg.AddEntry(eff_fd, 'Feed-down', 'lep')
    leg.Draw()

    #_____________________________________________________________________________________
    # Save output
    outfile_name = f'{outputdir}/eff_{charm_hadron}{suffix}.root'
    outfile = ROOT.TFile(outfile_name, 'RECREATE')
    outfile.cd()
    c.Write()
    eff_prompt.Write()
    eff_fd.Write()
    gen_prompt_hist.Write()
    gen_fd_hist.Write()
    rec_prompt_hist.Write()
    rec_fd_hist.Write()
    outfile.Close()
    input(f'Saving efficiency histograms to {outfile_name}. Press any key to exit.')

def compute_eff_thns(config_file, centclass, inputFile, outputdir, suffix, batch=False):

    gROOT.SetBatch(batch)

    #_____________________________________________________________________________________
    # Read configuration file
    with open(config_file, 'r', encoding='utf8') as ymlconfig:
        config = yaml.load(ymlconfig, yaml.FullLoader)

    #_____________________________________________________________________________________
    # Load input files
    charm_hadron = config['Dmeson']
    infile = ROOT.TFile.Open(inputFile)
    ptMins = config['ptmins']
    ptMaxs = config['ptmaxs']
    ptLims = list(ptMins)
    nPtBins = len(ptMins)
    ptLims.append(ptMaxs[-1])

    #_____________________________________________________________________________________
    # Check centrality selection
    centMin, centMax = get_centrality_bins(centclass)[1]
    check_cent_sel(charm_hadron, centMin, centMax, infile)

    #_____________________________________________________________________________________
    # define histograms
    hEffPrompt = TH1F('hEffPrompt', ';#it{p}_{T} (GeV/#it{c});Efficiency', nPtBins, np.asarray(ptLims, 'd'))
    hEffFD = TH1F('hEffFD', ';#it{p}_{T} (GeV/#it{c});Efficiency', nPtBins, np.asarray(ptLims, 'd'))
    hYieldPromptGen = TH1F('hYieldPromptGen', ';#it{p}_{T} (GeV/#it{c}); # Generated MC', nPtBins, np.asarray(ptLims, 'd'))
    hYieldFDGen = TH1F('hYieldFDGen', ';#it{p}_{T} (GeV/#it{c}); # Generated MC', nPtBins, np.asarray(ptLims, 'd'))
    hYieldPromptReco = TH1F('hYieldPromptReco', ';#it{p}_{T} (GeV/#it{c}); # Reco MC', nPtBins, np.asarray(ptLims, 'd'))
    hYieldFDReco = TH1F('hYieldFDReco', ';#it{p}_{T} (GeV/#it{c}); # Reco MC', nPtBins, np.asarray(ptLims, 'd'))
    SetObjectStyle(hEffPrompt, markerstyle=20,
                   markercolor=ROOT.kOrange+1,
                   markersize=1., linecolor=ROOT.kOrange+1)
    SetObjectStyle(hEffFD, markerstyle=21,
                   linestyle=2,
                   markercolor=ROOT.kAzure+2, markersize=1.,
                   linecolor=ROOT.kAzure+2)
    SetObjectStyle(hYieldPromptGen, color=ROOT.kRed+1, markerstyle=20)
    SetObjectStyle(hYieldFDGen, color=ROOT.kAzure+4, markerstyle=21, markersize=1.5, linewidh=2, linestyle=7)
    SetObjectStyle(hYieldPromptReco, color=ROOT.kRed+1, markerstyle=20)
    SetObjectStyle(hYieldFDReco, color=ROOT.kAzure+4, markerstyle=21, markersize=1.5, linewidh=2, linestyle=7)

    hRecoPrompt, hRecoFD, hGenPrompt, hGenFD = ([] for _ in range(4))

    #_____________________________________________________________________________________
    # Compute efficiency
    for iPt, (ptMin, ptMax) in enumerate(zip(ptMins, ptMaxs)):
        ## get inpput histograms
        ## whether need to minus reflection from prompt or FD?
        hRecoPrompt.append(infile.Get(f'cent_bins{centMin}_{centMax}/pt_bins%0.f_%0.f/hPromptPt' % (ptMin*10, ptMax*10)))
        hRecoFD.append(infile.Get(f'cent_bins{centMin}_{centMax}/pt_bins%0.f_%0.f/hFDPt' % (ptMin*10, ptMax*10)))
        hGenPrompt.append(infile.Get(f'cent_bins{centMin}_{centMax}/pt_bins%0.f_%0.f/hPromptGenPt' % (ptMin*10, ptMax*10)))
        hGenFD.append(infile.Get(f'cent_bins{centMin}_{centMax}/pt_bins%0.f_%0.f/hFDGenPt' % (ptMin*10, ptMax*10)))

        ## load the values
        nRecoPromptUnc, nGenPromptUnc, nRecoFDUnc, nGenFDUnc = (ctypes.c_double() for _ in range(4))
        nRecoPrompt = hRecoPrompt[iPt].IntegralAndError(0, hRecoPrompt[iPt].GetNbinsX()+1, nRecoPromptUnc)
        nGenPrompt = hGenPrompt[iPt].IntegralAndError(0, hGenPrompt[iPt].GetNbinsX()+1, nGenPromptUnc)
        nRecoFD = hRecoFD[iPt].IntegralAndError(0, hRecoFD[iPt].GetNbinsX()+1, nRecoFDUnc)
        nGenFD = hGenFD[iPt].IntegralAndError(0, hGenFD[iPt].GetNbinsX()+1, nGenFDUnc)

        ## calculate efficiency
        effPrompt, effPromptUnc = ComputeEfficiency(nRecoPrompt, nGenPrompt, nRecoPromptUnc.value, nGenPromptUnc.value)
        effFD, effFDUnc = ComputeEfficiency(nRecoFD, nGenFD, nRecoFDUnc.value, nGenFDUnc.value)
        hEffPrompt.SetBinContent(iPt+1, effPrompt)
        hEffPrompt.SetBinError(iPt+1, effPromptUnc)
        hEffFD.SetBinContent(iPt+1, effFD)
        hEffFD.SetBinError(iPt+1, effFDUnc)

        hYieldPromptGen.SetBinContent(iPt+1, nGenPrompt)
        hYieldPromptGen.SetBinError(iPt+1, nGenPromptUnc.value)
        hYieldFDGen.SetBinContent(iPt+1, nGenFD)
        hYieldFDGen.SetBinError(iPt+1, nGenFDUnc.value)
        hYieldPromptReco.SetBinContent(iPt+1, nRecoPrompt)
        hYieldPromptReco.SetBinError(iPt+1, nRecoPromptUnc.value)
        hYieldFDReco.SetBinContent(iPt+1, nRecoFD)
        hYieldFDReco.SetBinError(iPt+1, nRecoFDUnc.value)

    #_____________________________________________________________________________________
    # Draw histograms
    leg = TLegend(0.6, 0.2, 0.8, 0.4)
    leg.SetTextSize(0.045)
    leg.SetFillStyle(0)
    leg.AddEntry(hEffPrompt, "Prompt", "p")
    leg.AddEntry(hEffFD, "Feed-down", "p")

    cEff = TCanvas('cEff', '', 800, 800)
    cEff.DrawFrame(ptMins[0], 1.e-5, ptMaxs[nPtBins-1], 1.,
                ';#it{p}_{T} (GeV/#it{c});Efficiency;')
    cEff.SetLogy()
    hEffPrompt.Draw('same')
    hEffFD.Draw('same')
    leg.Draw()

    #_____________________________________________________________________________________
    # Save output
    os.makedirs(f'{outputdir}/eff', exist_ok=True)
    outFileName = f'{outputdir}/eff/eff_{suffix}.root'
    outFile = TFile(outFileName, 'recreate')
    hEffPrompt.Write()
    hEffFD.Write()
    hYieldPromptGen.Write()
    hYieldFDGen.Write()
    hYieldPromptReco.Write()
    hYieldFDReco.Write()
    outFile.Close()

    outFileNamePDF = outFileName.replace('.root', '.pdf')
    cEff.SaveAs(outFileNamePDF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="configuration file")
    parser.add_argument('infileName', metavar='text', help='projection file')
    parser.add_argument('--centclass', '-c', metavar='text', default='')
    parser.add_argument("--outputdir", "-o", metavar="text",
                        default=".", help="output directory")
    parser.add_argument("--suffix", "-s", metavar="text",
                        default="", help="suffix for output files")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="batch mode")
    args = parser.parse_args()

    if not args.infileName:
        compute_eff(
            config_file=args.config,
            centclass=args.centclass,
            outputdir=args.outputdir,
            suffix=args.suffix)
    else:
        compute_eff_thns(
            config_file=args.config,
            centclass=args.centclass,
            inputFile=args.infileName,
            outputdir=args.outputdir,
            suffix=args.suffix,
            batch=args.batch)