#Parameters
#----------

#- config (str): path of directory with config files
#- an_res_file (str): path of directory with analysis results
#- centrality (str): centrality class
#- resolution (str/int): resolution file or resolution value
#- outputdir (str): output directory
#- suffix (str): suffix for output files
#- vn_method (str): vn technique (sp, ep, deltaphi)
#- wagon_id (str): wagon ID
#- skip_resolution (bool): skip resolution extraction
#- skip_projection (bool): skip projection extraction
#- skip_vn (bool): skip raw yield extraction
#----------

# resutls will be saved in ${workdir}/Results/${dataCent}/${centrality}/${size}/
scriptdir=$(dirname $0)
workdir=path/to/DmesonAnalysis/run3/flow/
config=path/to/config_flow.yml

centrality=k3050 # k020 k3050 k6080  <-----------------------------------------------------------------------------------------------
dataCent=2060 # 020 2050 50100   <---------------------------------------------------------------------------------------------------
size=large # small medium large <-------------------------------------------------------------------------------------------------------
vn_method=sp # sp ep deltaphi   <---------------------------------------------------------------------------------------------------
qvec= # full recenter   <-------------------------------------------------------------------------------------------------------
debug= #_old #_new #_tot

wagon_id= # 13649 14351 13650 14352    <---------------------------------------------------------------------------------------
doReso=false # false true    <-------------------------------------------------------------------------------------------------------
doProj=false # false true

an_res_file="path/to/AnRes_0.root \
path/to/AnRes_1.root \
path/to/AnRes_2.root \
path/to/AnRes_3.root \
path/to/AnRes_4.root \
path/to/AnRes_5.root \
path/to/AnRes_6.root \
path/to/AnRes_7.root
"

resolution=path/to/resolution.root

if [ ! -z "$wagon_id" ]; then wagon="-w ${wagon_id}" ; else wagon="" ; fi

# suffix
cent="${centrality:1}"
siz="${size:0:1}"
qve="${qvec:0:2}"
suffix=${cent}${siz}_${qve}${debug}

if $doReso; then
    reso="--skip_projection --skip_vn"
    suffix=${suffix}_Reso
else 
    reso="-r ${resolution} --skip_resolution" 
fi

if $doProj; then
    proj="--skip_resolution --skip_vn"
    suffix=${suffix}_Proj
else
    proj=""
fi

# output dir.
outputdir=${workdir}/Results/${dataCent}/${centrality}/${size}/
if [ ! -d "${outputdir}" ]; then mkdir -p ${outputdir}; fi

    python3 ${scriptdir}/run_full_flow_analysis.py \
    ${config} \
    ${an_res_file} \
    -c ${centrality} \
    -o ${outputdir} \
    -s ${suffix} \
    -v ${vn_method} \
    ${reso} \
    ${wagon} \
    $proj \
    --skip_efficiency \
    #--skip_projection

if [ ! -z "$wagon_id" ]; then
    cp -rf ${outputdir}/${wagon_id}/*  ${outputdir}
    rm -rf ${outputdir}/${wagon_id}
fi

