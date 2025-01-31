'''
Analysis utilities for flow analysis
'''

import ROOT
import os
import sys
import ctypes
from itertools import combinations
import numpy as np

def get_vn_versus_mass(thnSparses, inv_mass_bins, mass_axis, vn_axis, debug=False):
    '''
    Project vn versus mass

    Input:
        - thnSparse:
            THnSparse, input THnSparse obeject (already projected in centrality and pt)
        - inv_mass_bins:
            list of floats, bin edges for the mass axis
        - mass_axis:
            int, axis number for mass
        - vn_axis:
            int, axis number for vn
        - debug:
            bool, if True, create a debug file with the projections (default: False)

    Output:
        - hist_mass_proj:
            TH1D, histogram with vn as a function of mass
    '''
    if not isinstance(thnSparses, list):
        thnSparses = [thnSparses]
        
    for iThn, thnSparse in enumerate(thnSparses):
        hist_vn_proj_temp = thnSparse.Projection(vn_axis, mass_axis)
        hist_vn_proj_temp.SetName(f'hist_vn_proj_{iThn}')
        hist_vn_proj_temp.SetDirectory(0)
        
        if iThn == 0:
            hist_vn_proj = hist_vn_proj_temp.Clone('hist_vn_proj')
            hist_vn_proj.SetDirectory(0)
            hist_vn_proj.Reset()
            
        hist_vn_proj.Add(hist_vn_proj_temp)

    hist_mass_proj = thnSparse.Projection(mass_axis)
    hist_mass_proj.Reset()
    invmass_bins = np.array(inv_mass_bins)
    hist_mass_proj = ROOT.TH1D('hist_mass_proj', 'hist_mass_proj', len(invmass_bins)-1, invmass_bins)

    if debug:
        outfile = ROOT.TFile('debug.root', 'RECREATE')

    for i in range(hist_mass_proj.GetNbinsX()):
        bin_low = hist_vn_proj.GetXaxis().FindBin(invmass_bins[i])
        bin_high = hist_vn_proj.GetXaxis().FindBin(invmass_bins[i+1])
        profile = hist_vn_proj.ProfileY(f'profile_{bin_low}_{bin_high}', bin_low, bin_high)
        mean_sp = profile.GetMean()
        mean_sp_err = profile.GetMeanError()
        hist_mass_proj.SetBinContent(i+1, mean_sp)
        hist_mass_proj.SetBinError(i+1, mean_sp_err)

    if debug:
        hist_vn_proj.Write()
        hist_mass_proj.Write()
        outfile.Close()

    return hist_mass_proj

def get_occupancy(thnSparses, occupancy_axis, debug=False):
    '''
    Project occupancy versus mass

    Input:
        - thnSparse:
            THnSparse, input THnSparse obeject (already projected in centrality and pt)
        - occupancy_axis:
            int, axis number for occupancy
        - debug:
            bool, if True, create a debug file with the projections (default: False)

    Output:
        - hist_occupancy:
            TH1D, histogram with vn as a function of mass
    '''
    if not isinstance(thnSparses, list):
        thnSparses = [thnSparses]
        
    for iThn, thnSparse in enumerate(thnSparses):
        hist_occupancy_temp = thnSparse.Projection(occupancy_axis)
        hist_occupancy_temp.SetName(f'hist_occupancy_{iThn}')
        hist_occupancy_temp.SetDirectory(0)
        
        if iThn == 0:
            hist_occupancy = hist_occupancy_temp.Clone('hist_occupancy')
            hist_occupancy.SetDirectory(0)
            hist_occupancy.Reset()
            
        hist_occupancy.Add(hist_occupancy_temp)
    # hist_occupancy = thnSparse.Projection(occupancy_axis)
    
    if debug:
        outfile = ROOT.TFile('debug.root', 'RECREATE')
        hist_occupancy.Write()
        outfile.Close()

    return hist_occupancy

def get_evselbits(thnSparse, evselbits_axis, debug=False):
    '''
    Project evselbits versus mass

    Input:
        - thnSparse:
            THnSparse, input THnSparse obeject (already projected in centrality and pt)
        - evselbits_axis:
            int, axis number for evselbits
        - debug:
            bool, if True, create a debug file with the projections (default: False)

    Output:
        - hist_evselbits:
            TH1D, histogram with vn as a function of mass
    '''
    if not isinstance(thnSparse, list):
        thnSparse = [thnSparse]
    
    for iThn, thnSparse in enumerate(thnSparse):
        hist_evselbits_temp = thnSparse.Projection(evselbits_axis)
        hist_evselbits_temp.SetName(f'hist_evselbits_{iThn}')
        hist_evselbits_temp.SetDirectory(0)
        
        if iThn == 0:
            hist_evselbits = hist_evselbits_temp.Clone('hist_evselbits')
            hist_evselbits.SetDirectory(0)
            hist_evselbits.Reset()
            
        hist_evselbits.Add(hist_evselbits_temp)
    
    # hist_evselbits = thnSparse.Projection(evselbits_axis)
    
    if debug:
        outfile = ROOT.TFile('debug.root', 'RECREATE')
        hist_evselbits.Write()
        outfile.Close()

    return hist_evselbits

def get_resolution(dets, det_lables, cent_min_max):
    '''
    Compute resolution for SP or EP method

    Input:
        - dets:
            list of TH2D, list of TH2D objects with the SP product or EP cos(deltaphi) values vs centrality
        - det_lables:
            list of strings, list of detector labels
        - cent_min_max:
            list of floats, max and min centrality bins

    Output:
        - histo_means:
            list of TH1D, list of histograms with the mean value of the projections as a function of centrality for 1% bins
        - histo_means_deltacent:
            list of TH1D, list of histograms with the mean value of the projections as a function of centrality for CentMin-CentMax
        - histo_reso:
            TH1D, histogram with the resolution value as a function of centrality for 1% bins
        - histo_reso_delta_cent:
            TH1D, histogram with the resolution value as a function of centrality for CentMin-CentMax
    '''
    histo_projs, histo_means, histo_means_deltacent = [], [], []

    # collect the qvecs and prepare histo for mean and resolution
    for _, (det, det_label) in enumerate(zip(dets, det_lables)):
        print(f'Processing {det_label}')
        # th1 for mean 1% centrality bins
        histo_means.append(ROOT.TH1F('', '', cent_min_max[1]-cent_min_max[0], cent_min_max[0], cent_min_max[1]))
        histo_means[-1].SetDirectory(0)
        histo_means[-1].SetName(f'proj_{det_label}_mean')
        # th1 for mean CentMin-CentMax
        histo_projs.append([])
        hist_proj_dummy = det.ProjectionY(f'proj_{det.GetName()}_mean_deltacent',
                                          det.GetXaxis().FindBin(cent_min_max[0]),
                                          det.GetXaxis().FindBin(cent_min_max[1])-1)
        histo_means_deltacent.append(ROOT.TH1F('', '', 1, cent_min_max[0], cent_min_max[1]))
        histo_means_deltacent[-1].SetDirectory(0)
        histo_means_deltacent[-1].SetName(f'proj_{det_label}_mean_deltacent')

        # Set mean values for CentMin-CentMax
        histo_means_deltacent[-1].SetBinContent(1, hist_proj_dummy.GetMean())
        histo_means_deltacent[-1].SetBinError(1, hist_proj_dummy.GetMeanError())
        del hist_proj_dummy

        # collect projections 1% centrality bins
        for cent in range(cent_min_max[0], cent_min_max[1]):
            bin_cent = det.GetXaxis().FindBin(cent) # common binning
            histo_projs[-1].append(det.ProjectionY(f'proj_{det_label}_{cent}',
                                                          bin_cent, bin_cent))
        # Set mean values for 1% centrality bins
        for ihist, _ in enumerate(histo_projs[-1]):
            histo_means[-1].SetBinContent(ihist+1, histo_projs[-1][ihist].GetMean())

    # Compute resolution for 1% centrality bins
    histo_reso = ROOT.TH1F('histo_reso', 'histo_reso',
                           cent_min_max[1]-cent_min_max[0],
                           cent_min_max[0], cent_min_max[1])
    histo_reso.SetDirectory(0)
    for icent in range(cent_min_max[0], cent_min_max[1]):
        reso = compute_resolution([histo_means[i].GetBinContent(icent-cent_min_max[0]+1) for i in range(len(dets))])
        centbin = histo_reso.GetXaxis().FindBin(icent)
        histo_reso.SetBinContent(centbin, reso)

    # Compute resolution for CentMin-CentMax
    histo_reso_delta_cent = ROOT.TH1F('histo_reso_delta_cent', 'histo_reso_delta_cent',
                                      1, cent_min_max[0], cent_min_max[1])
    res_deltacent = compute_resolution([histo_means_deltacent[i].GetBinContent(1) for i in range(len(dets))])
    histo_reso_delta_cent.SetBinContent(1, res_deltacent)
    histo_reso_delta_cent.SetDirectory(0)

    return histo_means, histo_means_deltacent, histo_reso, histo_reso_delta_cent

def getListOfHisots(an_res_file, wagon_id, vn_method):
    '''
    Get list of histograms for SP or EP resolution

    Input:
        - an_res_file:
            str, resolution file
        - wagon_id:
            str, wagon ID
        - vn_method:
            str, vn method (sp or ep)

    Output:
        - correct_histo_triplets:
            list of TH2D, list of TH2D objects with the SP product or EP cos(deltaphi) values vs centrality
        - correct_histo_labels:
            list of strings, list of detector labels
    '''
    infile_path = f'hf-task-flow-charm-hadrons'
    if wagon_id:
        infile_path = f'{infile_path}_id{wagon_id}'
    if vn_method != 'sp':
        infile_path = f'{infile_path}/{vn_method}Reso'
        prefix = f'hEpReso'
    else:
        infile_path = f'{infile_path}/spReso'
        prefix = 'hSpReso'

    infile = ROOT.TFile(an_res_file, 'READ')
    directory = infile.GetDirectory(infile_path)
    histos = [key.ReadObj() for key in directory.GetListOfKeys()]
    for histo in histos:
        histo.SetDirectory(0)
    pairs = [key.GetName() for key in directory.GetListOfKeys()]

    # generate triplets of pairs (AB, AC, BC)
    triplets = []
    detsA = ['FT0c', 'FT0a', 'FV0a', 'TPCpos', 'FT0m', 'TPCneg']
    triplets = list(combinations(pairs, 3))
    histo_triplets = list(combinations(histos, 3))
    correct_histo_triplets = []
    correct_histo_labels = []
    for i, triplet in enumerate(triplets):
        for detA in detsA:
            detB = triplet[0].replace(prefix, '').replace(detA, '')
            detC = triplet[1].replace(prefix, '').replace(detA, '')
            if (detA in triplet[0] and detA in triplet[1]) and \
               (detB in triplet[0] and detB in triplet[2]) and \
               (detC in triplet[1] and detC in triplet[2]):
                correct_histo_triplets.append(histo_triplets[i])
                correct_histo_labels.append((detA, detB, detC))

    return correct_histo_triplets, correct_histo_labels

def compute_resolution(subMean):
    '''
    Compute resolution for SP or EP method

    Input:
        - subMean:
            list of floats, list of mean values of the projections

    Output:
        - resolution:
            float, resolution value
    '''
    print(subMean)
    if len(subMean) == 1:
        resolution =  subMean[0]
        if resolution <= 0:
            return 0
        else:
            return np.sqrt(resolution)
    elif len(subMean) == 3:
        print('3 subsystems')
        resolution = (subMean[0] * subMean[1]) / subMean[2] if subMean[2] != 0 else 0
        if resolution <= 0:
            return 0
        else:
            print(resolution, np.sqrt(resolution))
            return np.sqrt(resolution)
    else:
        print('ERROR: dets must be a list of 2 or 3 subsystems')
        sys.exit(1)

def get_centrality_bins(centrality):
    '''
    Get centrality bins

    Input:
        - centrality:
            str, centrality class (e.g. 'k3050')

    Output:
        - cent_bins:
            list of floats, centrality bins
        - cent_label:
            str, centrality label
    '''
    if centrality == 'k010':
        return '0_10', [0, 10]
    if centrality == 'k020':
        return '0_20', [0, 20]
    if centrality == 'k2030':
        return '20_30', [20, 30]
    elif centrality == 'k3040':
        return '30_40', [30, 40]
    elif centrality == 'k3050':
        return '30_50', [30, 50]
    elif centrality == 'k4050':
        return '40_50', [40, 50]
    elif centrality == 'k2060':
        return '20_60', [20, 60]
    elif centrality == 'k4060':
        return '40_60', [40, 60]
    elif centrality == 'k6070':
        return '60_70', [60, 70]
    elif centrality == 'k6080':
        return '60_80', [60, 80]
    elif centrality == 'k7080':
        return '70_80', [70, 80]
    elif centrality == 'k0100':
        return '0_100', [0, 100]
    else:
        print(f"ERROR: cent class \'{centrality}\' is not supported! Exit")
    sys.exit()

def compute_r2(reso_file, wagon_id, cent_min, cent_max, detA, detB, detC, vn_method):
    '''
    Compute resolution for SP or EP method
    
    Input:
        - reso_file:
            TFile, resolution file
        - wagon_id:
            str, wagon ID
        - cent_min:
            int, minimum centrality bin
        - cent_max:
            int, maximum centrality bin
        - detA:
            str, detector A
        - detB:
            str, detector B
        - detC:
            str, detector C
        - do_ep:
            bool, if True, compute EP resolution
            if False, compute SP resolution

    Output:
        - reso:
            float, resolution value
    '''
    if wagon_id != '':
        wagon_id = f'{wagon_id}'
    if vn_method != 'sp':
        hist_name = f'hf-task-flow-charm-hadrons{wagon_id}/epReso/hEpReso'
    else:
        hist_name = f'hf-task-flow-charm-hadrons{wagon_id}/spReso/hSpReso'

    detA_detB = reso_file.Get(f'{hist_name}{detA}{detB}')
    detA_detC = reso_file.Get(f'{hist_name}{detA}{detC}')
    detB_detC = reso_file.Get(f'{hist_name}{detB}{detC}')

    cent_bin_min = detA_detB.GetXaxis().FindBin(cent_min)
    cent_bin_max = detA_detB.GetXaxis().FindBin(cent_max)

    proj_detA_detB = detA_detB.ProjectionY(f'{hist_name}{detA}{detB}_proj{cent_min}_{cent_max}',
                                           cent_bin_min, cent_bin_max)
    proj_detA_detC = detA_detC.ProjectionY(f'{hist_name}{detA}{detC}_proj{cent_min}_{cent_max}',
                                           cent_bin_min, cent_bin_max)
    proj_detB_detC = detB_detC.ProjectionY(f'{hist_name}{detB}{detC}_proj{cent_min}_{cent_max}',
                                           cent_bin_min, cent_bin_max)

    average_detA_detB = proj_detA_detB.GetMean()
    average_detA_detC = proj_detA_detC.GetMean()
    average_detB_detC = proj_detB_detC.GetMean()

    reso = (average_detA_detB * average_detA_detC) / average_detB_detC if average_detB_detC != 0 else -999
    reso = np.sqrt(reso) if reso > 0 else -999
    return reso

# TODO: extend to vn not only v2
def get_invmass_vs_deltaphi(thnSparses, deltaphiaxis, invmassaxis):
    '''
    Project invariant mass versus deltaphi
    
    Input:
        - thnSparse:
            THnSparse, input THnSparse obeject
        - deltaphiaxis:
            int, axis number for deltaphi
        - invmassaxis:
            int, axis number for invariant mass

    Output:
        - hist_invMass_in:
            TH1D, histogram with invariant mass for in-plane
        - hist_invMass_out:
            TH1D, histogram with invariant mass for out-of-plane
    ''' 
    if not isinstance(thnSparses, list):
        thnSparses = [thnSparses]
    
    for iThn, thnSparse in enumerate(thnSparses):
        thn_inplane = thnSparse.Clone(f'thn_inplane')
        thn_outplane = thnSparse.Clone(f'thn_outplane')
        hist_cosDeltaPhi_inplane_temp = thn_inplane.Projection(deltaphiaxis, invmassaxis)
        hist_cosDeltaPhi_outplane_temp = thn_outplane.Projection(deltaphiaxis, invmassaxis)
        # In-plane (|cos(deltaphi)| < pi/4)
        hist_cosDeltaPhi_inplane_temp.GetYaxis().SetRangeUser(0, 1)
        hist_invMass_in_temp = hist_cosDeltaPhi_inplane_temp.Clone(f'hist_invMass_in')
        # Out-of-plane (|cos(deltaphi)| > pi/4)
        hist_cosDeltaPhi_outplane_temp.GetYaxis().SetRangeUser(-1, 0)
        hist_invMass_out_temp = hist_cosDeltaPhi_outplane_temp.Clone(f'hist_invMass_out')
        if iThn == 0:
            hist_invMass_in = hist_invMass_in_temp.Clone('hist_invMass_in')
            hist_invMass_in.SetDirectory(0)
            hist_invMass_in.Reset()
            hist_invMass_out = hist_invMass_out_temp.Clone('hist_invMass_out')
            hist_invMass_out.SetDirectory(0)
            hist_invMass_out.Reset()
        hist_invMass_in.Add(hist_invMass_in_temp)
        hist_invMass_out.Add(hist_invMass_out_temp)
        hist_invMass_in.SetLineColor(ROOT.kRed)
        del thn_inplane, thn_outplane, hist_cosDeltaPhi_inplane_temp, hist_cosDeltaPhi_outplane_temp, hist_invMass_in_temp, hist_invMass_out_temp
    
    return hist_invMass_in, hist_invMass_out

def get_vnfitter_results(vnFitter, secPeak, useRefl):
    '''
    Get vn fitter results:
    0: BkgInt
    1: BkgSlope
    2: SgnInt
    3: Mean
    4: Sigma
    5: SecPeakInt
    6: SecPeakMean
    7: SecPeakSigma
    8: ConstVnBkg
    9: SlopeVnBkg
    10: v2Sgn
    11: v2SecPeak
    12: reflection

    Input:
        - vnfitter:
            VnVsMassFitter, vn fitter object
        - secPeak:
            bool, if True, save secondary peak results
        - useRefl:
            bool, if True, save the results with reflection

    Output:
        - vn_results:
            dict, dictionary with vn results
            vn: vn value
            vnUnc: uncertainty of vn value
            mean: mean value
            meanUnc: uncertainty of mean value
            sigma: sigma value
            sigmaUnc: uncertainty of sigma value
            ry: raw yield
            ryUnc: uncertainty of raw yield
            ryTrue: true raw yield
            ryTrueUnc: uncertainty of true raw yield
            signif: significance
            signifUnc: uncertainty of significance
            chi2: reduced chi2
            prob: fit probability
            fTotFuncMass: total fit function for mass
            fTotFuncVn: total fit function for vn
            secPeakMeanMass: secondary peak mean mass
            secPeakMeanMassUnc: uncertainty of secondary peak mean mass
            secPeakSigmaMass: secondary peak sigma mass
            secPeakSigmaMassUnc: uncertainty of secondary peak sigma mass
            secPeakMeanVn: secondary peak mean vn
            secPeakMeanVnUnc: uncertainty of secondary peak mean vn
            secPeakSigmaVn: secondary peak sigma vn
            secPeakSigmaVnUnc: uncertainty of secondary peak sigma vn
            vnSecPeak: vn secondary peak
            vnSecPeakUnc: uncertainty of vn secondary peak
            fMassRflFunc: mass reflection function
            fMassBkgRflFunc: mass background reflection function
    '''
    vn_results = {}
    vn_results['vn'] = vnFitter.GetVn()
    vn_results['vnUnc'] = vnFitter.GetVnUncertainty()
    vn_results['mean'] = vnFitter.GetMean()
    vn_results['meanUnc'] = vnFitter.GetMeanUncertainty()
    vn_results['sigma'] = vnFitter.GetSigma()
    vn_results['sigmaUnc'] = vnFitter.GetSigmaUncertainty()
    vn_results['ry'] = vnFitter.GetRawYield()
    vn_results['ryUnc'] = vnFitter.GetRawYieldUncertainty()
    vn_results['chi2'] = vnFitter.GetReducedChiSquare()
    vn_results['prob'] = vnFitter.GetFitProbability()
    vn_results['fTotFuncMass'] = vnFitter.GetMassTotFitFunc()
    vn_results['fTotFuncVn'] = vnFitter.GetVnVsMassTotFitFunc()
    vn_results['fBkgFuncMass'] = vnFitter.GetMassBkgFitFunc()
    vn_results['fBkgFuncVn'] = vnFitter.GetVnVsMassBkgFitFunc()
    vn_results['fSgnFuncMass'] = vnFitter.GetMassSignalFitFunc()
    vn_results['fMassTemplFuncts'] = vnFitter.GetMassTemplFuncts()
    vn_results['fVnTemplFuncts'] = vnFitter.GetVnTemplFuncts()
    bkg, bkgUnc = ctypes.c_double(), ctypes.c_double()
    vnFitter.Background(3, bkg, bkgUnc)
    vn_results['bkg'] = bkg.value
    vn_results['bkgUnc'] = bkgUnc.value
    sgn, sgnUnc = ctypes.c_double(), ctypes.c_double()
    vnFitter.Signal(3, sgn, sgnUnc)
    vn_results['ryTrue'] = sgn.value
    vn_results['ryTrueUnc'] = sgnUnc.value
    signif, signifUnc = ctypes.c_double(), ctypes.c_double()
    vnFitter.Significance(3, signif, signifUnc)
    vn_results['signif'] = signif.value
    vn_results['signifUnc'] = signifUnc.value

    if secPeak:
        vn_results['secPeakMeanMass'] = vn_results['fTotFuncMass'].GetParameter(vn_results['fTotFuncMass'].GetParName(6))
        vn_results['secPeakMeanMassUnc'] = vn_results['fTotFuncMass'].GetParError(6)
        vn_results['secPeakSigmaMass'] = vn_results['fTotFuncMass'].GetParameter(vn_results['fTotFuncMass'].GetParName(7))
        vn_results['secPeakSigmaMassUnc'] = vn_results['fTotFuncMass'].GetParError(7)
        vn_results['secPeakMeanVn'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(6))
        vn_results['secPeakMeanVnUnc'] = vn_results['fTotFuncVn'].GetParError(6)
        vn_results['secPeakSigmaVn'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(7))
        vn_results['secPeakSigmaVnUnc'] = vn_results['fTotFuncVn'].GetParError(7)
        vn_results['vnSecPeak'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(11))
        vn_results['vnSecPeakUnc'] = vn_results['fTotFuncVn'].GetParError(11)

    if useRefl:
        vn_results['fMassRflFunc'] = vnFitter.GetMassRflFunc()
        vn_results['fMassBkgRflFunc'] = vnFitter.GetMassBkgRflFunc()

    return vn_results

def get_ep_vn(harmonic, nIn, nInUnc, nOut, nOutUnc, resol=1, corr=0):
    '''
    Compute EP vn

    Input:
        - harmonic:
            int, harmonic number
        - nIn:
            float, number of in-plane particles
        - nInUnc:
            float, uncertainty of the number of in-plane particles
        - nOut:
            float, number of out-of-plane particles
        - nOutUnc:
            float, uncertainty of the number of out-of-plane particles
        - resol:
            float, resolution value (default: 1)
        - corr:
            float, correlation between nIn and nOut (default: 1)

    Output:
        - vn:
            float, vn value
        - vnunc:
            float, uncertainty of vn value
    '''
    print(corr)
    if nIn + nOut == 0:
        print('\033[91m ERROR: nIn + nOut = 0. Return 0, 0 \033[0m')
        return 0, 0
    anis = (nIn - nOut) / (nIn + nOut)
    anisDerivIn  = 2 * nOut / ((nIn + nOut)*(nIn + nOut))
    anisDerivOut = -2 * nIn / ((nIn + nOut)*(nIn + nOut))
    anisunc = anisDerivIn * anisDerivIn * nInUnc * nInUnc +\
              anisDerivOut * anisDerivOut * nOutUnc * nOutUnc + \
              2 * anisDerivIn * anisDerivOut * nInUnc * nOutUnc * corr
    if anisunc < 0:
        print('\033[91m ERROR: anisunc < 0. Return 0, 0 \033[0m')
        return 0, 0
    anisunc = np.sqrt(anisunc)

    vn = (np.pi * anis) / (harmonic * harmonic * resol)
    vnunc = (np.pi * anisunc) / (harmonic * harmonic * resol)

    return vn, vnunc

def check_file_exists(file_path):
    '''
    Check if file exists

    Input:
        - file_path:
            str, file path

    Output:
        - file_exists:
            bool, if True, file exists
    '''
    file_exists = False
    if os.path.exists(file_path):
        file_exists = True
    return file_exists

def check_histo_exists(file, histo_name):
    '''
    Check if histogram exists in file

    Input:
        - file:
            TFile, ROOT file
        - histo_name:
            str, histogram name

    Output:
        - histo_exists:
            bool, if True, histogram exists
    '''
    if not check_file_exists(file):
        return False
    file = ROOT.TFile(file, 'READ')
    histo_exists = False
    if file.Get(histo_name):
        histo_exists = True
    return histo_exists

def get_refl_histo(reflFile, ptMins, ptMaxs):
    '''
    Method that loads MC histograms for the reflections of D0

    Input:
        - reflFile:
           TFile, ROOT file, include reflections of D0
        - ptMins:
            list, min pt bins
        - ptMaxs:
            list, max pt bins
    
    Output:
        - useRefl:
            bool, if True, MC histograms for the reflections of D0 exists
        - hMCSgn:
            lsit, signal histograms of D0
        - hMCRefl:
            list, reflection histograms of D0
    '''
    hMCSgn, hMCRefl = [], []
    if not check_file_exists(reflFile):
        print(f'Error: reflection file {reflFile} does not exist! Turning off reflections usage')
        return False
    
    reflFile = ROOT.TFile(reflFile, 'READ')

    for iPt, (ptMin, ptMax) in enumerate(zip(ptMins, ptMaxs)):
        ptLowLabel = ptMin * 10
        ptHighLabel = ptMax * 10

        hMCSgn.append(reflFile.Get(f'hFDMass_{ptLowLabel:.0f}_{ptHighLabel:.0f}'))
        print(f'hFDMass_{ptLowLabel}_{ptHighLabel}')
        hMCSgn[iPt].Add(reflFile.Get(f'hPromptMass_{ptLowLabel:.0f}_{ptHighLabel:.0f}'))
        if hMCSgn[iPt] == None:
            print(f'hFDMass_{ptLowLabel:.0f}_{ptHighLabel:.0f} or hPromptMass_{ptLowLabel:.0f}_{ptHighLabel:.0f} not found! Turning off reflections usage')
            return False
        hMCSgn[iPt].SetName(f'histSgn_{iPt}')
        hMCSgn[iPt].SetDirectory(0)

        hMCRefl.append(reflFile.Get(f'hVarReflMass_{ptLowLabel:.0f}_{ptHighLabel:.0f}'))
        if hMCRefl[iPt] == None:
            print(f'hVarReflMass_{ptLowLabel:.0f}0_{ptHighLabel:.0f}0 not found! Turning off reflections usage')
            return False
        hMCRefl[iPt].SetName(f'histRfl_{iPt}')
        hMCRefl[iPt].SetDirectory(0)

    reflFile.Close()

    return True, hMCSgn, hMCRefl

import yaml

def get_particle_info(particleName):
    '''
    Get particle information

    Input:
        - particleName: 
            the name of the particle

    Output:
        - particleTit: 
            the title of the particle
        - massAxisTit: 
            the title of the mass axis
        - decay: 
            the decay of the particle
        - massForFit: 
            float, the mass of the particle
    '''

    if particleName == 'Dplus':
        particleTit = 'D^{+}'
        massAxisTit = '#it{M}(K#pi#pi) (GeV/#it{c}^{2})'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(411).Mass()
        decay = 'D^{+} #rightarrow K^{#minus}#pi^{+}#pi^{+}'
    elif particleName == 'Ds':
        particleTit = 'D_{s}^{+}'
        massAxisTit = '#it{M}(KK#pi) (GeV/#it{c}^{2})'
        decay = 'D_{s}^{+} #rightarrow #phi#pi^{+} #rightarrow K^{+}K^{#minus}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(431).Mass()
    elif particleName == 'LctopKpi':
        particleTit = '#Lambda_{c}^{+}'
        massAxisTit = '#it{M}(pK#pi) (GeV/#it{c}^{2})'
        decay = '#Lambda_{c}^{+} #rightarrow pK^{#minus}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(4122).Mass()
    elif particleName == 'LctopK0s':
        massAxisTit = '#it{M}(pK^{0}_{s}) (GeV/#it{c}^{2})'
        decay = '#Lambda_{c}^{+} #rightarrow pK^{0}_{s}'
        massForFit = 2.25 # please calfully check the mass of Lc->pK0s, it is constant
        # massForFit = ROOT.TDatabasePDG.Instance().GetParticle(4122).Mass()
    elif particleName == 'Dstar':
        particleTit = 'D^{*+}'
        massAxisTit = '#it{M}(K#pi#pi) - #it{M}(K#pi) (GeV/#it{c}^{2})'
        decay = 'D^{*+} #rightarrow D^{0}#pi^{+} #rightarrow K^{#minus}#pi^{+}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(413).Mass() - ROOT.TDatabasePDG.Instance().GetParticle(421).Mass()
    elif particleName == 'Dzero':
        particleTit = 'D^{0}'
        massAxisTit = '#it{M}(K#pi) (GeV/#it{c}^{2})'
        decay = 'D^{0} #rightarrow K^{#minus}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(421).Mass()
    else:
        print(f'ERROR: the particle "{particleName}" is not supported! Choose between Dzero, Dplus, Ds, Dstar, and Lc. Exit!')
        sys.exit()

    return particleTit, massAxisTit, decay, massForFit

def get_cut_sets(pt_mins, pt_maxs, sig_cut, bkg_cut_maxs, correlated_cuts=True):
    '''
    Get cut sets

    Input:
        - pt_mins:
            list of floats, list of minimum pt values
        - pt_maxs:
            list of floats, list of maximum pt values
        - sig_cut:
            list or dict, signal cut
        - bkg_cut_maxs:
            list of (floats or list of floats), list of maximum bkg cut
        - correlated_cuts:
            bool, if True, correlated cuts

    Output:
        - nCutSets:
            list of ints, number of cut sets
        - sig_cuts_lower:
            list of lists of floats, list of lower edge for signal cuts
        - sig_cuts_upper:
            list of lists of floats, list of upper edge for signal cuts
        - bkg_cuts_lower:
            list of lists of floats, list of lower edge for background cuts (0)
        - bkg_cuts_upper:
            list of lists of floats, list of upper edge for background cuts
    '''
    nCutSets = []
    sig_cuts_lower, sig_cuts_upper, bkg_cuts_lower, bkg_cuts_upper = {}, {}, {}, {}
    if correlated_cuts:
        sig_cut_mins = sig_cut['min']
        sig_cut_maxs = sig_cut['max']
        sig_cut_steps = sig_cut['step']

        # compute the signal cutsets for each pt bin
        sig_cuts_lower = [list(np.arange(sig_cut_mins[iPt], sig_cut_maxs[iPt], sig_cut_steps[iPt])) for iPt in range(len(pt_mins))]
        sig_cuts_upper = [[1.0 for _ in range(len(sig_cuts_lower[iPt]))] for iPt in range(len(pt_mins))]

        # compute the ncutsets by signal cut for each pt bin
        nCutSets = [len(sig_cuts_lower[iPt]) for iPt in range(len(pt_mins))]

        # bkg cuts lower edge should always be 0
        bkg_cuts_lower = [[0. for _ in range(nCutSets[iPt])] for iPt in range(len(pt_mins))]
        bkg_cuts_upper = [[bkg_cut_maxs[iPt] for _ in range(nCutSets[iPt])] for iPt in range(len(pt_mins))]

    else:
        # load the signal cut
        sig_cuts_lower = [sig_cut[iPt]['min'] for iPt in range(len(pt_mins))]
        sig_cuts_upper = [sig_cut[iPt]['max'] for iPt in range(len(pt_mins))]
        
        # compute the ncutsets by the signal cut for each pt bin
        nCutSets = [len(sig_cuts_lower[iPt]) for iPt in range(len(pt_mins))]
        
        # load the background cut
        bkg_cuts_lower = [[0. for _ in range(nCutSets[iPt])] for iPt in range(len(pt_mins))]
        bkg_cuts_upper = [bkg_cut_maxs[iPt] for iPt in range(len(pt_mins))]
        
    # safety check

    for iPt in range(len(pt_mins)):
        assert len(sig_cuts_lower[iPt]) == len(sig_cuts_upper[iPt]) == len(bkg_cuts_lower[iPt]) == len(bkg_cuts_upper[iPt]) == nCutSets[iPt], \
            f"Mismatch in lengths for pt bin {iPt}: {len(sig_cuts_lower[iPt])}, {len(sig_cuts_upper[iPt])}, \
            {len(bkg_cuts_lower[iPt])}, {len(bkg_cuts_upper[iPt])}, \
            {nCutSets[iPt]}"

    return nCutSets, sig_cuts_lower, sig_cuts_upper, bkg_cuts_lower, bkg_cuts_upper

def get_cut_sets_config(config):
    '''
    Get cut sets from configuration file
    Input:
        - config:
            str, flow configuration file
    Output:
        - nCutSets:
            list of ints, number of cut sets
        - sig_cuts_lower:
            list of lists of floats, list of lower edge for signal cuts
        - sig_cuts_upper:
            list of lists of floats, list of upper edge for signal cuts
        - bkg_cuts_lower:
            list of lists of floats, list of lower edge for background cuts (0)
        - bkg_cuts_upper:
            list of lists of floats, list of upper edge for background cuts
    '''
    with open(config, 'r') as ymlCfgFile:
        config = yaml.load(ymlCfgFile, yaml.FullLoader)

    ptmins = config['ptmins']
    ptmaxs = config['ptmaxs']
    correlated_cuts = config['minimisation']['correlated']
    if correlated_cuts:
        sig_cut = config['cut_variation']['corr_bdt_cut']['sig']
        bkg_cut_maxs = config['cut_variation']['corr_bdt_cut']['bkg_max']
    else:
        sig_cut = config['cut_variation']['uncorr_bdt_cut']['sig']
        bkg_cut_maxs = config['cut_variation']['uncorr_bdt_cut']['bkg_max']

    return get_cut_sets(ptmins, ptmaxs, sig_cut, bkg_cut_maxs, correlated_cuts)