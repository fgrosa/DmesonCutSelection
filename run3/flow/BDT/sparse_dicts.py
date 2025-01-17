import ROOT
from ROOT import TFile

def get_sparses(config):
    
    axes_dict = {}
    
    infileflow = ROOT.TFile(config['flow_filename'])
    sparseFlow = infileflow.Get('hf-task-flow-charm-hadrons/hSparseFlowCharm')
    infileflow.Close()
    axes_dict['Flow'] = {
        'Mass': 0,
        'Pt': 1,
        'cent': 2,
        'sp': 3,
        'score_bkg': 4,
        'score_FD': 5,
        'occ': 6
    }
        
    print(f"infiletaskpath: {config['eff_filename']}")
    infiletask = ROOT.TFile(config['eff_filename'])
    sparsesReco, sparsesGen = {}, {}    
    if config['Dmeson'] == 'Dzero':
        axes_reco = {
            'score_bkg': 0,
            'score_prompt': 2,
            'score_FD': 1,
            'Mass': 3,
            'Pt': 4,
            'y': 5,
            'cand_type': 6,
            'pt_bmoth': 7,
            'origin': 8,
            'npvcontr': 9,
            'cent': 10,
            'occ': 11,
        }
        axes_gen = {
            'Pt': 0,
            'pt_bmoth': 1,
            'y': 2,
            'origin': 3,
            'cent': 4,
            'occ': 5
        }
        print(f"infiletask: {infiletask}")
        sparsesReco['RecoPrompt'] = infiletask.Get('hf-task-d0/hBdtScoreVsMassVsPtVsPtBVsYVsOriginVsD0Type')
        print(f"sparsesReco['RecoPrompt']: {sparsesReco['RecoPrompt']}")
        print('\n')
        sparsesReco['RecoPrompt'].GetAxis(axes_reco['origin']).SetRange(1, 2)  # make sure it is signal
        sparsesReco['RecoPrompt'].GetAxis(axes_reco['cand_type']).SetRange(2, 2) # make sure it is prompt
        axes_dict['RecoPrompt'] = axes_reco
        sparsesReco['RecoFD'] = infiletask.Get('hf-task-d0/hBdtScoreVsMassVsPtVsPtBVsYVsOriginVsD0Type')
        sparsesReco['RecoFD'].GetAxis(axes_reco['cand_type']).SetRange(3, 3)  # make sure it is non-prompt
        sparsesReco['RecoFD'].GetAxis(axes_reco['origin']).SetRange(1, 2)  # make sure it is signal
        axes_dict['RecoFD'] = axes_reco
        sparsesReco['RecoRefl'] = infiletask.Get('hf-task-d0/hBdtScoreVsMassVsPtVsPtBVsYVsOriginVsD0Type')
        sparsesReco['RecoRefl'].GetAxis(axes_reco['origin']).SetRange(3, 4)  # make sure it is reflection
        axes_dict['RecoRefl'] = axes_reco
        sparsesReco['RecoReflPrompt'] = infiletask.Get('hf-task-d0/hBdtScoreVsMassVsPtVsPtBVsYVsOriginVsD0Type')
        sparsesReco['RecoReflPrompt'].GetAxis(axes_reco['origin']).SetRange(3, 4)  # make sure it is reflection
        sparsesReco['RecoReflPrompt'].GetAxis(axes_reco['cand_type']).SetRange(2, 2) # make sure it is prompt
        axes_dict['RecoReflPrompt'] = axes_reco
        sparsesReco['RecoReflFD'] = infiletask.Get('hf-task-d0/hBdtScoreVsMassVsPtVsPtBVsYVsOriginVsD0Type')
        sparsesReco['RecoReflFD'].GetAxis(axes_reco['origin']).SetRange(3, 4)  # make sure it is reflection
        sparsesReco['RecoReflFD'].GetAxis(axes_reco['cand_type']).SetRange(3, 3) # make sure it is FD
        axes_dict['RecoReflFD'] = axes_reco
        sparsesGen['GenPrompt'] = infiletask.Get('hf-task-d0/hSparseAcc')
        sparsesGen['GenPrompt'].GetAxis(axes_gen['origin']).SetRange(2, 2)  # make sure it is prompt
        axes_dict['GenPrompt'] = axes_gen 
        print(f"sparseGenPrompt: {sparsesGen['GenPrompt']}")
        sparsesGen['GenFD'] = infiletask.Get('hf-task-d0/hSparseAcc')
        sparsesGen['GenFD'].GetAxis(axes_gen['origin']).SetRange(3, 3)  # make sure it is non-prompt
        axes_dict['GenFD'] = axes_gen
        print(f"sparseGenFD: {sparsesGen['GenFD']}")
        #TODO: safety checks for Dmeson reflecton and secondary peak
    elif config['Dmeson'] == 'Dplus':
        sparsesReco['RecoPrompt'] = infiletask.Get('hf-task-dplus/hSparseMassPrompt')
        axes_dict['RecoPrompt'] = {
            'Mass': 0,
            'Pt': 1,
            'score_bkg': 2,
            'score_prompt': 3,
            'score_FD': 4,
            'cent': 5,
            'occ': 6,
        }
        sparsesReco['RecoFD'] = infiletask.Get('hf-task-dplus/hSparseMassFD')
        axes_dict['RecoFD'] = {
            'Mass': 0,
            'Pt': 1,
            'pt_bmoth': 2,
            'flag_bhad': 3,
            'score_bkg': 4,
            'score_prompt': 5,
            'score_FD': 6,
            'cent': 7,
            'occ': 8
        }
        sparsesGen['GenPrompt'] = infiletask.Get('hf-task-dplus/hSparseMassGenPrompt')
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'y': 1,
            'cent': 2,
            'occ': 3
        }   
        print(f"sparseGenPrompt: {sparsesGen['GenPrompt']}")
        sparsesGen['GenFD'] = infiletask.Get('hf-task-dplus/hSparseMassGenFD')
        axes_dict['GenFD'] = {
            'Pt': 0,
            'y': 1,
            'pt_bmoth': 2,
            'flag_bhad': 3,
            'cent': 4,
            'occ': 5
        }
        print(f"sparseGenFD: {sparsesGen['GenFD']}")
    elif config['Dmeson'] == 'Ds':
        sparsesReco['RecoPrompt'] = infiletask.Get('hf-task-ds/MC/Ds/Prompt/hSparseMass')
        axes_dict['RecoPrompt'] = {
            'Mass': 0,
            'Pt': 1,
            'cent': 3,
            'npvcontr': 4,
            'score_bkg': 5,
            'score_prompt': 6,
            'score_FD': 7,
            'occ': 8,
        }
        sparsesReco['RecoFD'] = infiletask.Get('hf-task-ds/MC/Ds/NonPrompt/hSparseMass')
        axes_dict['RecoFD'] = {
            'Mass': 0,
            'Pt': 1,
            'cent': 2,
            'score_bkg': 3,
            'score_prompt': 4,
            'score_FD': 5,
            'pt_bmoth': 6,
            'flag_bhad': 7,
            'occ': 8
        }
        sparsesGen['GenPrompt'] = infiletask.Get('hf-task-ds/MC/Ds/Prompt/hSparseGen')
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'y': 1,
            'npvcontr': 2,
            'cent': 3,
            'occ': 4
        }   
        print(f"sparseGenPrompt: {sparsesGen['GenPrompt']}")
        sparsesGen['GenFD'] = infiletask.Get('hf-task-ds/MC/Ds/NonPrompt/hSparseGen')
        axes_dict['GenFD'] = {
            'Pt': 0,
            'y': 1,
            'cent': 2,
            'pt_bmoth': 3,
            'flag_bhad': 4,
            'occ': 5
        }
        print(f"sparseGenFD: {sparsesGen['GenFD']}")

    infiletask.Close()
    return sparseFlow, sparsesReco, sparsesGen, axes_dict
