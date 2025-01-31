'''
Script to project the MC distributions and apply the pt weights from the AnRes.root of Dtask
python3 proj_thn_mc.py config_flow.yml config_cutset.yml -o path/to/output -s text
                                                        --ptweights path/to/file histName 
                                                        --ptweightsB path/to/file histName
'''
import ROOT
import uproot
import yaml
import argparse
import sys
import os
from ROOT import gROOT
from alive_progress import alive_bar
from scipy.interpolate import InterpolatedUnivariateSpline
sys.path.append('../../..')
from utils.TaskFileLoader import LoadSparseFromTask



def proj_MC(config, cutsetConfig, ptweights, ptweightsB, outputdir, suffix):

    with open(config, 'r') as ymlCfgFile:
        config = yaml.load(ymlCfgFile, yaml.FullLoader)

    # read configuration
    inputFiles = config['MC_filename']
    if not isinstance(inputFiles, list):
        inputFiles = [inputFiles]
    particleName = config['Dmeson']
    enableRef = config['enableRef']
    enableSecPeak = config['enableSecPeak']
    Bspeciesweights = config['Bspeciesweights'] if 'Bspeciesweights' in config else None

    #TODO: safety checks for Dmeson reflecton and secondary peak

    # check the arguments
    if ptweights == []:
        print('\033[91m WARNING: pt weights will not be provided! \033[0m')
        ptweights = None
    if ptweightsB == []:
        print('\033[91m WARNING: B weights will not not be provided! \033[0m')
        ptweightsB = None
    if Bspeciesweights == None:
        print('\033[91m WARNING: B species weights will not be provided! \033[0m')

    # load thnsparse
    for iFile, inputFile in enumerate(inputFiles):
        if iFile == 0:
            sparseReco, sparseGen = LoadSparseFromTask(inputFile, config, True)
        else:
            sparseRecoPart, sparseGenPart = LoadSparseFromTask(inputFile, config, True)
            for sparsetype in sparseRecoPart:
                sparseReco[sparsetype].Add(sparseRecoPart[sparsetype])
            for sparsetype in sparseGenPart:
                sparseGen[sparsetype].Add(sparseGenPart[sparsetype])

    refSparse = 'RecoPrompt'

    # compute the pt weights
    if ptweights:
        ptWeights = uproot.open(ptweights[0])[ptweights[1]]
        bins = ptWeights.axis(0).edges()
        ptCentW = [(bins[iBin]+bins[iBin+1])/2 for iBin in range(len(bins)-1)]
        sPtWeights = InterpolatedUnivariateSpline(ptCentW, ptWeights.values())

    if ptweightsB:
        ptWeightsB = uproot.open(ptweightsB[0])[ptweightsB[1]]
        bins = ptWeightsB.axis(0).edges()
        ptCentWB = [(bins[iBin]+bins[iBin+1])/2 for iBin in range(len(bins)-1)]
        sPtWeightsB = InterpolatedUnivariateSpline(ptCentWB, ptWeightsB.values())

    with open(cutsetConfig, 'r') as ymlCutSetFile:
        cutSetCfg = yaml.load(ymlCutSetFile, yaml.FullLoader)
    cutVars = cutSetCfg['cutvars']

    # dictionaris of TH1
    all_dict = {'InvMass': [], 'Pt': []}
    prompt_dict = {'InvMass': [], 'Pt': []}
    fd_dict = {'InvMass': [], 'Pt': []}
    prompt_gen_list = []
    fd_gen_list = []
    refl_dict = {'InvMass': [], 'Pt': []}
    prompt_dict_secpeak = {'InvMass': [], 'Pt': []}
    fd_dict_secpeak = {'InvMass': [], 'Pt': []}
    prompt_gen_list_secpeak = []
    fd_gen_list_secpeak = []
    
    # output file
    os.makedirs(f'{outputdir}/proj_mc', exist_ok=True)
    outfile = ROOT.TFile(f'{outputdir}/proj_mc/proj_mc_{suffix}.root', 'recreate')

    with alive_bar(len(cutVars['Pt']['min']), title='Processing pT bins') as bar:
        for iPt, (ptMin, ptMax) in enumerate(zip(cutVars['Pt']['min'], cutVars['Pt']['max'])):
            print(f'Projecting distributions for {ptMin:.1f} < pT < {ptMax:.1f} GeV/c')
            ptLowLabel = ptMin * 10
            ptHighLabel = ptMax * 10

            # apply pt weights for reconstruction level
            ## apply cuts
            for iVar in cutVars:
                if iVar == 'InvMass':
                    continue
                axisNum = cutVars[iVar]['axisnum']
                binMin = sparseReco[refSparse].GetAxis(axisNum).FindBin(cutVars[iVar]['min'][iPt] * 1.0001)
                binMax = sparseReco[refSparse].GetAxis(axisNum).FindBin(cutVars[iVar]['max'][iPt] * 0.9999)

                if iVar == 'ML_output_FD' or iVar == 'ML_output_Bkg':
                    print(f'{iVar}: {cutVars[iVar]["min"][iPt]} < {iVar} < {cutVars[iVar]["max"][iPt]}')

                if 'RecoAll' in sparseReco:
                    sparseReco['RecoAll'].GetAxis(axisNum).SetRange(binMin, binMax)

                if particleName == 'Dzero':
                    sparseReco['RecoPrompt'].GetAxis(8).SetRange(2, 2) # make sure it is prompt
                    sparseReco['RecoFD'].GetAxis(8).SetRange(3, 3)  # make sure it is non-prompt
                    sparseReco['RecoPrompt'].GetAxis(6).SetRange(1, 2)  # make sure it is signal
                    sparseReco['RecoFD'].GetAxis(6).SetRange(1, 2)  # make sure it is signal
                #TODO: add other particles
                sparseReco['RecoPrompt'].GetAxis(axisNum).SetRange(binMin, binMax)
                sparseReco['RecoFD'].GetAxis(axisNum).SetRange(binMin, binMax)

                if enableRef:
                    sparseReco['RecoRefl'].GetAxis(6).SetRange(3, 4)  # make sure it is reflection
                    sparseReco['RecoRefl'].GetAxis(axisNum).SetRange(binMin, binMax)
                    sparseReco['RecoReflPrompt'].GetAxis(6).SetRange(3, 4)  # make sure it is reflection
                    sparseReco['RecoReflPrompt'].GetAxis(8).SetRange(2, 2)  # make sure it is prompt reflection
                    sparseReco['RecoReflPrompt'].GetAxis(axisNum).SetRange(binMin, binMax)
                    sparseReco['RecoReflFD'].GetAxis(6).SetRange(3, 4)  # make sure it is reflection
                    sparseReco['RecoReflFD'].GetAxis(8).SetRange(3, 3)  # make sure it is non-prompt reflection
                    sparseReco['RecoReflFD'].GetAxis(axisNum).SetRange(binMin, binMax)

                if enableSecPeak:
                    sparseReco['RecoSecPeakPrompt'].GetAxis(axisNum).SetRange(binMin, binMax)
                    sparseReco['RecoSecPeakFD'].GetAxis(axisNum).SetRange(binMin, binMax)

            ## apply pt weights
            for iVar in ('InvMass', 'Pt'):
                hVarRefl, hVarReflPrompt, hVarReflFD, hVarPrompt, hVarFD = None, None, None, None, None
                varName = 'Pt' if iVar == 'Pt' else 'Mass'
                axisNum = cutVars[iVar]['axisnum']
                
                if 'RecoAll' in sparseReco:
                    hVar = sparseReco['RecoAll'].Projection(axisNum)
                    hVar.SetName(f'h{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                    outfile.cd()
                    all_dict[iVar].append(hVar)
                    hVar.Write()

                if iVar != 'Pt':
                    hVarPrompt = sparseReco['RecoPrompt'].Projection(axisNum)
                    hVarPrompt.SetName(f'hPrompt{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                    hVarFD = sparseReco['RecoFD'].Projection(axisNum)
                    hVarFD.SetName(f'hFD{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                else:
                    ### no pt weights
                    if not ptweights and not ptweightsB:
                        hVarPrompt = sparseReco['RecoPrompt'].Projection(axisNum)
                        hVarFD = sparseReco['RecoFD'].Projection(axisNum)

                    ### pt weights for prompt
                    if ptweights:
                        hVarPrompt = sparseReco['RecoPrompt'].Projection(axisNum)
                        for iBin in range(1, hVarPrompt.GetNbinsX()+1):
                            if hVarPrompt.GetBinContent(iBin) > 0.:
                                relStatUnc = hVarPrompt.GetBinError(iBin) / hVarPrompt.GetBinContent(iBin)
                                ptCent = hVarPrompt.GetBinCenter(iBin)
                                hVarPrompt.SetBinContent(iBin, hVarPrompt.GetBinContent(iBin) * sPtWeights(ptCent))
                                hVarPrompt.SetBinError(iBin, hVarPrompt.GetBinContent(iBin) * relStatUnc)
                    
                    ### pt weights for non-prompt, but no B species weights
                    if ptweightsB and not Bspeciesweights:
                        hPtBvsPtD = sparseReco['RecoFD'].Projection(2, axisNum)
                        for iPtD in range(1, hPtBvsPtD.GetXaxis().GetNbins()+1):
                            for iPtB in range(1, hPtBvsPtD.GetYaxis().GetNbins()+1):
                                ptCentB = hPtBvsPtD.GetYaxis().GetBinCenter(iPtB)
                                origContent = hPtBvsPtD.GetBinContent(iPtD, iPtB)
                                origError = hPtBvsPtD.GetBinError(iPtD, iPtB)
                                weight = 0
                                if sPtWeightsB(ptCentB) > 0:
                                    weight = sPtWeightsB(ptCentB)
                                content = origContent * weight
                                error = 0
                                if origContent > 0:
                                    error = origError / origContent * content
                                hPtBvsPtD.SetBinContent(iPtD, iPtB, content)
                                hPtBvsPtD.SetBinError(iPtD, iPtB, error)
                        hVarFD = hPtBvsPtD.ProjectionX(f'hFD{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}',
                                                        0, hPtBvsPtD.GetYaxis().GetNbins()+1, 'e')
                    ### pt weights from B for non-prompt and B species weights
                    elif ptweightsB and Bspeciesweights:
                        hPtBvsBspecievsPtD = sparseReco['RecoFD'].Projection(axisNum, 3, 2)
                        for iPtD in range(1, hPtBvsBspecievsPtD.GetXaxis().GetNbins()+1):
                            for iBspecie in range(1, hPtBvsBspecievsPtD.GetYaxis().GetNbins()+1):
                                for iPtB in range(1, hPtBvsBspecievsPtD.GetZaxis().GetNbins()+1):
                                    ptCentB = hPtBvsBspecievsPtD.GetZaxis().GetBinCenter(iPtB)
                                    origContent = hPtBvsBspecievsPtD.GetBinContent(iPtD, iBspecie, iPtB)
                                    origError = hPtBvsBspecievsPtD.GetBinError(iPtD, iBspecie, iPtB)
                                    weight = Bspeciesweights[iBspecie-1]
                                    if sPtWeightsB(ptCentB) > 0:
                                        weight *= sPtWeightsB(ptCentB)
                                    content = origContent * weight
                                    error = 0
                                    if origContent > 0:
                                        error = origError / origContent * content
                                    hPtBvsBspecievsPtD.SetBinContent(iPtD, iBspecie, iPtB, content)
                                    hPtBvsBspecievsPtD.SetBinError(iPtD, iBspecie, iPtB, error)
                        hVarFD = hPtBvsBspecievsPtD.ProjectionX(f'hFD{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}',
                                                                0, hPtBvsBspecievsPtD.GetYaxis().GetNbins()+1,
                                                                0, hPtBvsBspecievsPtD.GetZaxis().GetNbins()+1, 'e')
                    ### only B species weights
                    elif Bspeciesweights:
                        hBspecievsPtD = sparseReco['RecoFD'].Projection(3, axisNum)
                        for iPtD in range(1, hBspecievsPtD.GetXaxis().GetNbins()+1):
                            for iBspecie in range(1, hBspecievsPtD.GetYaxis().GetNbins()+1):
                                origContent = hBspecievsPtD.GetBinContent(iPtD, iBspecie)
                                origError = hBspecievsPtD.GetBinError(iPtD, iBspecie)
                                weight = Bspeciesweights[iBspecie-1]
                                content = origContent * weight
                                error = 0
                                if origContent > 0:
                                    error = origError / origContent * content
                                hBspecievsPtD.SetBinContent(iPtD, iBspecie, content)
                                hBspecievsPtD.SetBinError(iPtD, iBspecie, error)
                        hVarFD = hBspecievsPtD.ProjectionX(f'hFD{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}',
                                                        0, hBspecievsPtD.GetYaxis().GetNbins()+1, 'e')
                    ### use the pt weights from prompt for non-prompt
                    elif ptweights: # if pt weights for prompt are present apply them
                        for iBin in range(1, hVarFD.GetNbinsX()+1):
                                if hVarFD.GetBinContent(iBin) > 0.:
                                    relStatUnc = hVarFD.GetBinError(iBin) / hVarFD.GetBinContent(iBin)
                                    ptCent = hVarFD.GetBinCenter(iBin)
                                    hVarFD.SetBinContent(iBin, hVarFD.GetBinContent(iBin) * sPtWeights(ptCent))
                                    hVarFD.SetBinError(iBin, hVarFD.GetBinContent(iBin) * relStatUnc)

                ## wirte teh output           
                hVarPrompt.SetName(f'hPrompt{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                prompt_dict[iVar].append(hVarPrompt)
                hVarPrompt.Write()
                hVarFD.SetName(f'hFD{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                fd_dict[iVar].append(hVarFD)
                hVarFD.Write()
                if enableRef:
                    hVarRefl = sparseReco['RecoRefl'].Projection(axisNum)
                    hVarRefl.SetName(f'hVarRefl{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                    refl_dict[iVar].append(hVarRefl)
                    hVarRefl.Write()
                    hVarReflPrompt = sparseReco['RecoReflPrompt'].Projection(axisNum)
                    hVarReflPrompt.SetName(f'hVarReflPrompt{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                    refl_dict[iVar].append(hVarReflPrompt)
                    hVarReflPrompt.Write()
                    hVarReflFD = sparseReco['RecoReflFD'].Projection(axisNum)
                    hVarReflFD.SetName(f'hVarReflFD{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                    refl_dict[iVar].append(hVarReflFD)
                    hVarReflFD.Write()
                if enableSecPeak:
                    hVarPromptSecPeak = sparseReco['RecoSecPeakPrompt'].Projection(axisNum)
                    hVarPromptSecPeak.SetName(f'hPromptSecPeak{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                    prompt_dict_secpeak[iVar].append(hVarPromptSecPeak)
                    hVarPromptSecPeak.Write()
                    hVarFDSecPeak = sparseReco['RecoSecPeakFD'].Projection(axisNum)
                    hVarFDSecPeak.SetName(f'hFDSecPeak{varName}_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                    fd_dict_secpeak[iVar].append(hVarFDSecPeak)
                    hVarFDSecPeak.Write()
            # end of apply pt weights for reconstruction level

            # apply pt weights for generation level
            ## pt bins
            binGenMin = sparseGen['GenPrompt'].GetAxis(0).FindBin(ptMin*1.0001)
            binGenMax = sparseGen['GenPrompt'].GetAxis(0).FindBin(ptMax*0.9999)
            if particleName == 'Dzero':
                sparseGen['GenPrompt'].GetAxis(3).SetRange(2, 2)  # make sure it is prompt
                sparseGen['GenFD'].GetAxis(3).SetRange(3, 3)  # make sure it is non-prompt
            sparseGen['GenPrompt'].GetAxis(0).SetRange(binGenMin, binGenMax)
            sparseGen['GenFD'].GetAxis(0).SetRange(binGenMin, binGenMax)

            ## no pt weights
            if not ptweights and not ptweightsB:
                hGenPtPrompt = sparseGen['GenPrompt'].Projection(0)
                hGenPtFD = sparseGen['GenFD'].Projection(0)

            ## apply pt weights for prompt
            if ptweights:
                hGenPtPrompt = sparseGen['GenPrompt'].Projection(0)
                for iBin in range(1, hGenPtPrompt.GetNbinsX()+1):
                    if hGenPtPrompt.GetBinContent(iBin) > 0:
                        relStatUnc = hGenPtPrompt.GetBinError(iBin) / hGenPtPrompt.GetBinContent(iBin)
                        ptCent = hGenPtPrompt.GetBinCenter(iBin)
                        hGenPtPrompt.SetBinContent(iBin, hGenPtPrompt.GetBinContent(iBin) * sPtWeights(ptCent))
                        hGenPtPrompt.SetBinError(iBin, hGenPtPrompt.GetBinContent(iBin) * relStatUnc)

            ## apply pt weights for non-prompt
            ### pt weights from B for non-prompt, but no B species weights
            if ptweightsB and not Bspeciesweights:
                hPtBvsPtGenD = sparseGen['GenFD'].Projection(2, 0)
                for iPtD in range(1, hPtBvsPtGenD.GetXaxis().GetNbins()+1):
                    for iPtB in range(1, hPtBvsPtGenD.GetYaxis().GetNbins()+1):
                        ptCentB = hPtBvsPtGenD.GetYaxis().GetBinCenter(iPtB)
                        origContent = hPtBvsPtGenD.GetBinContent(iPtD, iPtB)
                        origError = hPtBvsPtGenD.GetBinError(iPtD, iPtB)
                        weight = 0
                        if sPtWeightsB(ptCent) > 0:
                            weight = sPtWeightsB(ptCentB)
                        content = hPtBvsPtGenD.GetBinContent(iPtD, iPtB) * weight
                        error = 0
                        if origContent > 0:
                            error = origError / origContent * content
                        hPtBvsPtGenD.SetBinContent(iPtD, iPtB, content)
                        hPtBvsPtGenD.SetBinError(iPtD, iPtB, error)
                hGenPtFD = hPtBvsPtGenD.ProjectionX(f'hFDGenPt_{ptLowLabel:.0f}_{ptHighLabel:.0f}',
                                                    0, hPtBvsPtGenD.GetYaxis().GetNbins()+1, 'e')
            ### pt weights from B for non-prompt and B species weights
            elif ptweightsB and Bspeciesweights:
                hPtBvsBspecievsPtGenD = sparseGen['GenFD'].Projection(0, 3, 2)
                for iPtD in range(1, hPtBvsBspecievsPtGenD.GetXaxis().GetNbins()+1):
                    for iBspecie in range(1, hPtBvsBspecievsPtGenD.GetYaxis().GetNbins()+1):
                        for iPtB in range(1, hPtBvsBspecievsPtGenD.GetZaxis().GetNbins()+1):
                            ptCentB = hPtBvsBspecievsPtGenD.GetZaxis().GetBinCenter(iPtB)
                            origContent = hPtBvsBspecievsPtGenD.GetBinContent(iPtD, iBspecie, iPtB)
                            origError = hPtBvsBspecievsPtGenD.GetBinError(iPtD, iBspecie, iPtB)
                            weight = Bspeciesweights[iBspecie-1]
                            if sPtWeightsB(ptCentB) > 0:
                                weight *= sPtWeightsB(ptCentB)
                            content = origContent * weight
                            error = 0
                            if origContent > 0:
                                error = origError / origContent * content
                            hPtBvsBspecievsPtGenD.SetBinContent(iPtD, iBspecie, iPtB, content)
                            hPtBvsBspecievsPtGenD.SetBinError(iPtD, iBspecie, iPtB, error)
                hGenPtFD = hPtBvsBspecievsPtGenD.ProjectionX(f'hFDGenPt_{ptLowLabel:.0f}_{ptHighLabel:.0f}',
                                                            0, hPtBvsBspecievsPtGenD.GetYaxis().GetNbins()+1,
                                                            0, hPtBvsBspecievsPtGenD.GetZaxis().GetNbins()+1, 'e')
            ### only B species weights
            elif Bspeciesweights:
                hBspecievsPtGenD = sparseGen['GenFD'].Projection(3, 0)
                for iPtD in range(1, hBspecievsPtGenD.GetXaxis().GetNbins()+1):
                    for iBspecie in range(1, hBspecievsPtGenD.GetYaxis().GetNbins()+1):
                        origContent = hBspecievsPtGenD.GetBinContent(iPtD, iBspecie)
                        origError = hBspecievsPtGenD.GetBinError(iPtD, iBspecie)
                        weight = Bspeciesweights[iBspecie-1]
                        content = origContent * weight
                        error = 0
                        if origContent > 0:
                            error = origError / origContent * content
                        hBspecievsPtGenD.SetBinContent(iPtD, iBspecie, content)
                        hBspecievsPtGenD.SetBinError(iPtD, iBspecie, error)
                hGenPtFD = hBspecievsPtGenD.ProjectionX(f'hFDGenPt_{ptLowLabel:.0f}_{ptHighLabel:.0f}',
                                                        0, hBspecievsPtGenD.GetYaxis().GetNbins()+1, 'e')
            ### use the pt weights from prompt for non-prompt
            elif ptweights: # if pt weights for prompt are present apply them
                hGenPtFD = sparseGen['GenFD'].Projection(0)
                            
            ## print the output
            hGenPtPrompt.SetName(f'hPromptGenPt_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
            prompt_gen_list.append(hGenPtPrompt)
            hGenPtPrompt.Write()
            hGenPtFD.SetName(f'hFDGenPt_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
            fd_gen_list.append(hGenPtFD)
            hGenPtFD.Write()
            if enableSecPeak:
                sparseGen['GenSecPeakPrompt'].GetAxis(0).SetRange(binGenMin, binGenMax)
                sparseGen['GenSecPeakFD'].GetAxis(0).SetRange(binGenMin, binGenMax)
                hGenPtPromptSecPeak = sparseGen['GenSecPeakPrompt'].Projection(0)
                hGenPtPromptSecPeak.SetName(f'hPromptSecPeakGenPt_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                prompt_gen_list_secpeak.append(hGenPtPromptSecPeak)
                hGenPtPromptSecPeak.Write()
                hGenPtFDSecPeak = sparseGen['GenSecPeakFD'].Projection(0)
                hGenPtFDSecPeak.SetName(f'hFDSecPeakGenPt_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
                fd_gen_list_secpeak.append(hGenPtFDSecPeak)
                hGenPtFDSecPeak.Write()

            bar()

        for iVar in cutVars:
            axisNum = cutVars[iVar]['axisnum']
            if 'RecoAll' in sparseReco:
                sparseReco['RecoAll'].GetAxis(axisNum).SetRange(-1, -1)
            sparseReco['RecoPrompt'].GetAxis(axisNum).SetRange(-1, -1)
            sparseReco['RecoFD'].GetAxis(axisNum).SetRange(-1, -1)
            if enableRef:
                sparseReco['RecoRefl'].GetAxis(axisNum).SetRange(-1, -1)
                sparseReco['RecoReflPrompt'].GetAxis(axisNum).SetRange(-1, -1)
                sparseReco['RecoReflFD'].GetAxis(axisNum).SetRange(-1, -1)
            if enableSecPeak:
                sparseReco['RecoSecPeakPrompt'].GetAxis(axisNum).SetRange(-1, -1)
                sparseReco['RecoSecPeakFD'].GetAxis(axisNum).SetRange(-1, -1)

    infile = ROOT.TFile(inputFiles[0])
    if particleName in ['Dplus', 'Ds']:
        cent_hist = infile.Get('hf-candidate-creator-3prong/hSelCollisionsCent')
        outfile_dir = 'hf-candidate-creator-3prong'
    elif particleName == 'Dzero':
        cent_hist = infile.Get('hf-candidate-creator-2prong/hSelCollisionsCent')
        outfile_dir = 'hf-candidate-creator-2prong'
    outfile.mkdir(outfile_dir)
    outfile.cd(outfile_dir)
    cent_hist.Write()

    infile.Close()
    outfile.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    parser.add_argument('cutsetConfig', metavar='text',
                        default='cutsetConfig.yaml', help='cutset configuration file')
    parser.add_argument("--ptweights", "-w", metavar="text", nargs=2, required=False,
                        default=[], help="path to pt weights file and histogram name")
    parser.add_argument("--ptweightsB", "-wb", metavar="text", nargs=2, required=False,
                        default=[], help="path to pt weightsB file and histogram name")
    parser.add_argument("--outputdir", "-o", metavar="text",
                        default=".", help="output directory")
    parser.add_argument("--suffix", "-s", metavar="text",
                        default="", help="suffix for output files")
    args = parser.parse_args()

    proj_MC(config=args.config,
            cutsetConfig=args.cutsetConfig,
            ptweights=args.ptweights,
            ptweightsB=args.ptweightsB,
            outputdir=args.outputdir,
            suffix=args.suffix)