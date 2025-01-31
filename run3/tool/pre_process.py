'''
This sricpt is used to pre-process a/multi large AnRes.root for the BDT training:
    - split the input by pT
    - obtain the sigma from prompt enhance sample
python3 pre_process.py config_pre.yml AnRes_1.root AnRes_2.root --pre --sigma  
'''
import os
import sys
import yaml
import array
import ROOT
import argparse
import concurrent.futures

def cook_thnsparse(thnsparse_list, ptmins, ptmaxs, axestokeep):
    '''
    Split the input THnSparse by pT bins and project onto axes to keep.

    Input:
        - thnsparse_list (list): List of input THnSparse objects.
        - ptmins (list): List of minimum pT values for each bin.
        - ptmaxs (list): List of maximum pT values for each bin.
        - axestokeep (list): List of axes to keep in the projection.

    Returns:
        - dict: Dictionary of projected THnSparse objects for each pT bin.
    '''
    thnsparses = {}
    for iThn, thnsparse in enumerate(thnsparse_list):
        #TODO: add possibility to apply cuts for different variables
        for iPt in range(0, len(ptmins)):
            binMin = thnsparse.GetAxis(1).FindBin(ptmins[iPt]*1.00001)
            binMax = thnsparse.GetAxis(1).FindBin(ptmaxs[iPt]*0.99999)
            thnsparse.GetAxis(1).SetRange(binMin, binMax)
            thn_proj = thnsparse.Projection(len(axestokeep), array.array('i', axestokeep), 'O')
            
            if iThn == 0:
                thnsparses[iPt] = thn_proj
            else:
                thnsparses[iPt].Add(thn_proj)
    return thnsparses

def pre_process(an_res_file, ptmins, ptmaxs, axestokeep, outputDir):
    
    # Load the ThnSparse
    thnsparse_list = []
    for file in an_res_file:
        infile = ROOT.TFile(file, 'READ')
        thnsparse_list.append(infile.Get('hf-task-flow-charm-hadrons/hSparseFlowCharm'))
        print(infile.GetName())

    def process_pt_bin(iPt, ptmin, ptmax, thnsparse_list, axestokeep, outputDir):
        pre_thnsparse = None
        # add posibility to apply cuts for different variables
        for iThn, thnsparse in enumerate(thnsparse_list):
            binMin = thnsparse.GetAxis(1).FindBin(ptmin * 1.00001)
            binMax = thnsparse.GetAxis(1).FindBin(ptmax * 0.99999)
            thnsparse.GetAxis(1).SetRange(binMin, binMax)
            
            thn_proj = thnsparse.Projection(len(axestokeep), array.array('i', axestokeep), 'O')
            thn_proj.SetName(thnsparse.GetName())
            
            if iThn == 0:
                pre_thnsparse = thn_proj.Clone()
            else:
                pre_thnsparse.Add(thn_proj)
        
        os.makedirs(f'{outputDir}/pre/AnRes', exist_ok=True)
        outFile = ROOT.TFile(f'{outputDir}/pre/AnRes/AnalysisResults_pt{iPt}.root', 'RECREATE')
        outFile.mkdir('hf-task-flow-charm-hadrons')
        outFile.cd('hf-task-flow-charm-hadrons')
        pre_thnsparse.Write()
        outFile.Close()
        
        del pre_thnsparse
        
        print(f'Finished processing pT bin {ptmin} - {ptmax}')

    # Loop over each pt bin in parallel
    max_workers = 12 # hyperparameter
    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        tasks = [executor.submit(process_pt_bin, iPt, ptmin, ptmax, thnsparse_list, axestokeep, outputDir) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]
        for task in concurrent.futures.as_completed(tasks):
            task.result()
        
def get_sigma(preFiles, config_pre, centrality, resolution, outputDir, skip_projection=False):
    with open(config_pre, 'r') as cfgPre:
        config = yaml.safe_load(cfgPre)

    apply_btd_cuts = config['apply_btd_cuts']    
    
    if not apply_btd_cuts:
        print("Warning: 'apply_btd_cuts' is not enabled in the config_pre.")
    
    if not skip_projection:
        skip_proj = ''
    else:
        skip_proj = '--skip_projection'
    
    os.makedirs(f'{outputDir}/pre/sigma', exist_ok=True)

    str_preFiles = ' '.join(preFiles)
    command = (f"python3 ../flow/run_full_flow_analysis.py {config_pre} {str_preFiles} \
                -c {centrality} -o {outputDir}/pre/sigma \
                -s sigma -v sp --r {resolution} \
                --skip_efficiency --skip_resolution \
                {skip_proj}")
    os.system(f'{command}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('config_pre', metavar='text', 
                        default='config_pre.yml', help='configuration file')
    parser.add_argument('an_res_file', metavar='text', 
                        nargs='+', help='input ROOT files with anres')
    parser.add_argument('--pre', action='store_true', help='pre-process the AnRes.root')
    parser.add_argument('--sigma', action='store_true', help='get the sigma')
    parser.add_argument('--skip_projection', '-sp', action='store_true', help='skip the projection')
    args = parser.parse_args()

    if not args.pre and not args.sigma:
        print('Please specify the action to perform.')
        sys.exit(1)

    with open(args.config_pre, 'r') as cfgPre:
        config = yaml.safe_load(cfgPre)
  
    # Load the configuration
    ptmins = config['ptmins']
    ptmaxs = config['ptmaxs']
    axestokeep = config['axestokeep']
    outputDir = config['outputDir']
    
    if args.pre:
        pre_process(args.an_res_file, ptmins, ptmaxs, axestokeep, outputDir)
    
    if args.sigma:
        
        centrality = config['centrality']
        resolution = config['resolution']
        
        if os.path.exists(f'{outputDir}/pre'):
            preFiles = [f'{outputDir}/pre/AnRes/AnalysisResults_pt{iFile}.root' for iFile in range(len(ptmins))]
        else:
            raise ValueError(f'No eff fodel found in {outputDir}')
        preFiles.sort()
        
        # you have to know the sigma from the differet prompt enhance samples is stable first
        get_sigma(preFiles, args.config_pre, centrality, resolution, outputDir, skip_projection=args.skip_projection)