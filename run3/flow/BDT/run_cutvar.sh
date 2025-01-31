#!/bin/bash

#Parameters
#----------
#- config (str): path of directory with config files
#- an_res_file (str): path of directory with analysis results
#- centrality (str): centrality class
#- resolution (str/int): resolution file or resolution value
#- outputdir (str): output directory
#- suffix (str): suffix for output files
#- vn_method (str): vn technique (sp, ep, deltaphi)
#- skip_calc_weights (bool): skip calculation of weights
#- skip_make_yaml (bool): skip make yaml
#- skip_cut_variation (bool): skip cut variation
#- skip_proj_mc (bool): skip projection for MC
#- skip_efficiency (bool): skip efficiency
#- skip_vn (bool): skip vn extraction
#- skip_frac_cut_var (bool): skip fraction by cut variation
#- skip_data_driven_frac (bool): skip fraction by data-driven method
#- skip_v2_vs_frac (bool): skip v2 vs FD fraction
#----------
export config_flow="path/to/config_flow.yml"
export anres_dir="path/to/AnRes1.root \
path/to/AnRes2.root \
path/to/AnRes3.root \
path/to/AnRes4.root \
path/to/AnRes5.root"
export output_dir="path/to/output" # full/corrected full/uncorrected
export cent="k3050"
export vn_method="sp"
export res_file="path/to/resolution.root"
export suffix="pt2_3" # _ will be added automatically

export spw=False # True or False (skip calculation of weights)
export smy=False # True or False (skip make yaml)
export scv=False # True or False (skip cut variation)
export spm=False # True or False (skip projection for MC)
export seff=False # True or False (skip efficiency)
export svn=False # True or False (skip vn extraction)
export sfcv=False # True or False (skip fraction by cut variation)
export sddf=False # True or False (skip fraction by data-driven method)
export sv2vf=False # True or False (skip v2 vs fraction)


# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${output_dir}/cutvar_${suffix}/logs"
LOG_FILE="${output_dir}/cutvar_${suffix}/logs/log_${TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Starting run_cutvar.sh at $(date)"
echo "Logging to: ${LOG_FILE}"


if [ $spw = False ]; then
	export skip_calc_weights=""
else
	export skip_calc_weights="--skip_calc_weights"
fi

if [ $smy = False ]; then
	export skip_make_yaml=""
else
	export skip_make_yaml="--skip_make_yaml"
fi

if [ $scv = False ]; then
	export skip_cut_variation=""
else
	export skip_cut_variation="--skip_cut_variation"
fi

if [ $spm = False ]; then
	export skip_proj_mc=""
else
	export skip_proj_mc="--skip_proj_mc"
fi

if [ $seff = False ]; then
	export skip_efficiency=""
else
	export skip_efficiency="--skip_efficiency"
fi

if [ $svn = False ]; then
	export skip_vn=""
else
	export skip_vn="--skip_vn"
fi

if [ $sfcv = False ]; then
	export skip_frac_cut_var=""
else
	export skip_frac_cut_var="--skip_frac_cut_var"
fi

if [ $sddf = False ]; then
	export skip_data_driven_frac=""
else
	export skip_data_driven_frac="--skip_data_driven_frac"
fi

if [ $sv2vf = False ]; then
	export skip_v2_vs_frac=""
else
	export skip_v2_vs_frac="--skip_v2_vs_frac"
fi

python3 run_cutvar.py $config_flow $anres_dir -c $cent -r $res_file -o $output_dir -s $suffix -vn $vn_method \
						$skip_calc_weights \
						$skip_make_yaml \
						$skip_cut_variation \
						$skip_proj_mc \
						$skip_efficiency \
						$skip_vn \
						$skip_frac_cut_var \
						$skip_data_driven_frac \
						$skip_v2_vs_frac

echo "Completed run_cutvar.sh at $(date)"
echo "Log saved to: ${LOG_FILE}"

python3 /home/wuct/ALICE/local/DmesonAnalysis/run3/tool/clean_logs.py $LOG_FILE