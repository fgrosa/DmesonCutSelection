import os
import sys
import numpy as np
import argparse
import yaml
import shutil
sys.path.append('..')
from flow_analysis_utils import get_cut_sets_config

def check_dir(dir):

	if not os.path.exists(dir):
		print(f"\033[32m{dir} does not exist, it will be created\033[0m")
		os.makedirs(dir)
	else:
		print(f"\033[33m{dir} already exists, it will be removed and recreat\033[0m")
		shutil.rmtree(dir)
		os.makedirs(dir)

	return

def run_full_cut_variation(config_flow, anres_dir, cent, res_file, output, suffix, vn_method, 
						   skip_calc_weights=False,
						   skip_make_yaml=False, 
						   skip_cut_variation=False,
						   skip_proj_mc=False,
						   skip_efficiency=False,
						   skip_vn = False,
						   skip_frac_cut_var=False,
						   skip_data_driven_frac=False,
						   skip_v2_vs_frac=False):

#___________________________________________________________________________________________________________________________
	# Load and copy the configuration file
	with open(config_flow, 'r') as cfgFlow:
		config = yaml.safe_load(cfgFlow)
	
	CutSets, _, _, _, _ = get_cut_sets_config(config_flow)
	nCutSets = max(CutSets)

	print(f"\033[32mNumber of cutsets: {nCutSets}\033[0m")

	output_dir = f"{output}/cutvar_{suffix}"
 
	os.system(f"mkdir -p {output_dir}")

	# the pT weights histograms
	if 'ptWeights_path' in config and config['ptWeights_path'] is not None:
		given_ptweights = True
		given_ptWeightsPath = config['ptWeights_path']
	else:
		given_ptweights = False
  
	# copy the configuration file
	config_suffix = 0
	os.makedirs(f'{output_dir}/config_flow', exist_ok=True)
	while os.path.exists(f'{output_dir}/config_flow/config_flow_{suffix}_{config_suffix}.yml'):
		config_suffix = config_suffix + 1
	os.system(f'cp {config_flow} {output_dir}/config_flow/config_flow_{suffix}_{config_suffix}.yml')

	# backup the results into history
	file_to_check = f"{output_dir}/V2VsFrac/V2VsFrac_{suffix}.root"
	if os.path.exists(file_to_check):
		for sub_path in ['ry', 'CutVarFrac', 'V2VsFrac']:
			os.system(f"mkdir -p {output_dir}/history/{config_suffix}/{sub_path}")
			os.system(f"cp {output_dir}/{sub_path}/* {output_dir}/history/{config_suffix}/{sub_path}")
		os.system(f"cp {output_dir}/config_flow/config_flow_{suffix}_{config_suffix-1}.yml {output_dir}/history/{config_suffix}")

#___________________________________________________________________________________________________________________________
	# calculate the pT weights
	if not skip_calc_weights:
		check_dir(f"{output_dir}/ptweights")
		CalcWeiPath = "./ComputePtWeights.py"

		print(f"\033[32mpython3 {CalcWeiPath} {config_flow} -o {output_dir} -s {suffix}\033[0m")
		os.system(f"python3 {CalcWeiPath} {config_flow} -o {output_dir} -s {suffix}")
	else:
		print("\033[33mWARNING: Calculation of weights will not be performed\033[0m")

#___________________________________________________________________________________________________________________________
	# make yaml file
	if not skip_make_yaml:
		check_dir(f"{output_dir}/config")
		MakeyamlPath = './make_yaml_for_ml.py'

		print(f"\033[32mpython3 {MakeyamlPath} {config_flow} -o {output_dir} -s {suffix}\033[0m")
		os.system(f"python3 {MakeyamlPath} {config_flow} -o {output_dir} -s {suffix}")
	else:
		print("\033[33mWARNING: Make yaml will not be performed\033[0m")
	#TODO: 1.keep the yaml file for the user to check 2.modify the proj_thn_mc 3.use make_combination in proj_thn_mc.py

#___________________________________________________________________________________________________________________________
	# Cut variation (aply the cut and project)
	if not skip_cut_variation:
		check_dir(f"{output_dir}/proj")
		CutVarPath = "./cut_variation.py"

		anres_files = " ".join(anres_dir)
		print(f"\033[32mpython3 {CutVarPath} {config_flow} {anres_files} -c {cent} -r {res_file} -o {output_dir} -s {suffix}\033[0m")
		os.system(f"python3 {CutVarPath} {config_flow} {anres_files} -c {cent} -r {res_file} -o {output_dir} -s {suffix}")
	else:
		print("\033[33mWARNING: Cut variation will not be performed\033[0m")

#___________________________________________________________________________________________________________________________
	# Projection for MC and apply the ptweights
	if not skip_proj_mc:
		check_dir(f"{output_dir}/proj_mc")
		ProjMcPath = "./proj_thn_mc.py"
		
		if not os.path.exists(f'{output_dir}/ptweights/pTweight_{suffix}.root') and not given_ptweights:
			for i in range(nCutSets):
				iCutSets = f"{i:02d}"
				print(f"\033[32mpython3 {ProjMcPath} {config_flow} {output_dir}/config/cutset_{suffix}_{iCutSets}.yml -o {output_dir} -s {suffix}_{iCutSets}\033[0m")
				os.system(f"python3 {ProjMcPath} {config_flow} {output_dir}/config/cutset_{suffix}_{iCutSets}.yml -o {output_dir} -s {suffix}_{iCutSets}")
		else:
			for i in range(nCutSets):
				iCutSets = f"{i:02d}"

				if given_ptweights:
					ptweightsPath = given_ptWeightsPath
				else:
					ptweightsPath = f'{output_dir}/ptweights/pTweight_{suffix}.root'

				print(
					f"\033[32mpython3 {ProjMcPath} {config_flow} {output_dir}/config/cutset_{suffix}_{iCutSets}.yml "
					f"-w {ptweightsPath} hPtWeightsFONLLtimesTAMUDcent "
					f"-wb {ptweightsPath} hPtWeightsFONLLtimesTAMUBcent "
					f"-o {output_dir} -s {suffix}_{iCutSets} \033[0m"
				)
				os.system(f"python3 {ProjMcPath} {config_flow} {output_dir}/config/cutset_{suffix}_{iCutSets}.yml "
						f"-w {ptweightsPath} hPtWeightsFONLLtimesTAMUDcent "
						f"-wb {ptweightsPath} hPtWeightsFONLLtimesTAMUBcent "
						f"-o {output_dir} -s {suffix}_{iCutSets}")
	else:
		print("\033[33mWARNING: Projection for MC will not be performed\033[0m")							

#___________________________________________________________________________________________________________________________
	# Compute the efficiency
	if not skip_efficiency:
		check_dir(f"{output_dir}/eff")
		EffPath = "./../compute_efficiency.py"

		for i in range(nCutSets):
			iCutSets = f"{i:02d}"
			print(f"\033[32mpython3 {EffPath} {config_flow} {output_dir}/proj_mc/proj_mc_{suffix}_{iCutSets}.root -c {cent} -o {output_dir} -s {suffix}_{iCutSets}\033[0m")
			print(f"\033[32mProcessing cutset {iCutSets}\033[0m")
			os.system(f"python3 {EffPath} {config_flow} {output_dir}/proj_mc/proj_mc_{suffix}_{iCutSets}.root -c {cent} -o {output_dir} -s {suffix}_{iCutSets} --batch")
	else:
		print("\033[33mWARNING: Efficiency will not be performed\033[0m")

#___________________________________________________________________________________________________________________________
	# do the simulation fit to get the raw yields
	if not skip_vn:
		check_dir(f"{output_dir}/ry")
		SimFitPath = "./../get_vn_vs_mass.py"

		for i in range(nCutSets):
			iCutSets = f"{i:02d}"
			print(f"\033[32mpython3 {SimFitPath} {config_flow} {cent} {output_dir}/proj/proj_{suffix}.root -o {output_dir}/ry -s _{suffix}_{iCutSets} -vn {vn_method}\033[0m")
			print(f"\033[32mProcessing cutset {iCutSets}\033[0m")
			os.system(f"python3 {SimFitPath} {config_flow} {cent} {output_dir}/proj/proj_{suffix}_{iCutSets}.root -o {output_dir}/ry -s _{suffix}_{iCutSets} -vn {vn_method} --batch")
	else:
		print("\033[33mWARNING: vn extraction will not be performed\033[0m")

#___________________________________________________________________________________________________________________________
	# Compute the fraction by cut variation method
	if not skip_frac_cut_var:
		check_dir(f"{output_dir}/CutVarFrac")
		CurVarFracPath = "./compute_frac_cut_var.py"

		print(f"\033[32mpython3 {CurVarFracPath} {config_flow} {output_dir} -o {output_dir} -s {suffix}\033[0m")
		os.system(f"python3 {CurVarFracPath} {config_flow} {output_dir} -o {output_dir} -s {suffix} --batch")
	else:
		print("\033[33mWARNING: Fraction by cut variation will not be performed\033[0m")

#___________________________________________________________________________________________________________________________
	# Compute fraction by Data-driven method
	if not skip_data_driven_frac:
		check_dir(f"{output_dir}/DataDrivenFrac")
		DataDrivenFracPath = "./ComputeDataDriFrac_flow.py"

		print(f"\033[32mpython3 {DataDrivenFracPath} -i {output_dir} -o {output_dir} -s {suffix}\033[0m")
		os.system(f"python3 {DataDrivenFracPath} -i {output_dir} -o {output_dir} -s {suffix} --batch")
	else:
		print("\033[33mWARNING: Fraction by Data-driven method will not be performed\033[0m")

#___________________________________________________________________________________________________________________________
	# Compute v2 vs fraction
	if not skip_v2_vs_frac:
		check_dir(f"{output_dir}/V2VsFrac")
		v2vsFDFracPath = "./ComputeV2vsFDFrac.py"

		print(f"\033[32mpython3 {v2vsFDFracPath} {config_flow} -i {output_dir} -o {output_dir} -s {suffix}\033[0m")
		os.system(f"python3 {v2vsFDFracPath} {config_flow} -i {output_dir} -o {output_dir} -s {suffix}")
	else:
		print("\033[33mWARNING: v2 vs fraction will not be performed\033[0m")
	
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('flow_config', metavar='text', default='config_flow_d0.yml', help='configuration file')
	parser.add_argument('anres_dir', metavar='text', nargs='+', help='input ROOT files with anres')
	parser.add_argument("--centrality", "-c", metavar="text",default="k3050", help="centrality class")
	parser.add_argument("--resolution", "-r",  default="", help="resolution file/value")
	parser.add_argument("--outputdir", "-o", metavar="text", default=".", help="output directory")
	parser.add_argument("--suffix", "-s", metavar="text", default="", help="suffix for output files")
	parser.add_argument("--vn_method", "-vn", metavar="text", default="sp", help="vn technique (sp, ep, deltaphi)")
	parser.add_argument("--skip_calc_weights", "-scw", action="store_true", help="skip calculation of weights")
	parser.add_argument("--skip_make_yaml", "-smy", action="store_true", help="skip make yaml")
	parser.add_argument("--skip_cut_variation", "-scv", action="store_true", help="skip cut variation")
	parser.add_argument("--skip_proj_mc", "-spm", action="store_true", help="skip projection for MC")
	parser.add_argument("--skip_efficiency", "-se", action="store_true", help="skip efficiency")
	parser.add_argument("--skip_vn", "-svn", action="store_true", help="skip vn extraction")
	parser.add_argument("--skip_frac_cut_var", "-sf", action="store_true", help="skip fraction by cut variation")
	parser.add_argument("--skip_data_driven_frac", "-sddf", action="store_true", help="skip fraction by data-driven method")
	parser.add_argument("--skip_v2_vs_frac", "-sv2fd", action="store_true", help="skip v2 vs FD fraction")
	args = parser.parse_args()

	run_full_cut_variation(args.flow_config, args.anres_dir, args.centrality, args.resolution, args.outputdir, args.suffix, args.vn_method, 
						args.skip_calc_weights,
						args.skip_make_yaml, 
						args.skip_cut_variation, 
						args.skip_proj_mc, 
						args.skip_efficiency, 
						args.skip_vn,
						args.skip_frac_cut_var, 
						args.skip_data_driven_frac, 
						args.skip_v2_vs_frac)