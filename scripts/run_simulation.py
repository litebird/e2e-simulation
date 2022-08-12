import numpy as np
import subprocess
import sys

#command line example:
#python run_simulation.py 0 LFT L2-050

#parameters
isim           = sys.argv[1].zfill(3)
telescope      = sys.argv[2] #e.g. 'LFT'
channel        = sys.argv[3] #e.g. 'L2-050'
det_names_file = 'detectors_'+telescope+'_'+channel+'_T+B'#'detectors_HFT_H3-402_T+B'#.txt
nside          = 512
start_time     = '2030-04-01T00:00:00'
nproc          = 1#365    #number of simulation days, too
mapmaking_type = 'binned' #binned, destriper or all
imo_version    = 'v1.3'

#paths
coderoot        = #COMPLETE HERE   #folder where e2e_simulation.py is stored
base_path       = #COMPLETE HERE   #folder where you want to save the output files
input_maps_path = '/global/cfs/cdirs/litebird/simulations/maps/post_ptep_inputs_20220522/beam_convolved/'
user_email      = #COMPLETE HERE   #your email for notification



#create TOML file
toml_filename = 'e2e_sim'+isim+'_'+det_names_file+'_params'
with open(coderoot+'../ancillary/'+toml_filename+'.toml', 'w') as f:
    f.write('[general]\n')
    f.write('imo_version = \''+imo_version+'\'\n')
    f.write('input_maps_path = \''+input_maps_path+'\'\n')
    f.write('telescope = \''+telescope+'\'\n')
    f.write('det_names_file = \''+det_names_file+'\'\n')
    f.write('nside = '+str(nside)+'\n')
    f.write('isim = '+str(int(isim))+'\n')
    f.write('mission_time_days = \''+str(nproc)+'\'\n')
    f.write('mapmaking_type = \''+mapmaking_type+'\'\n')
    f.write('[simulation]\n')
    f.write('base_path = \''+base_path+'\'\n')
    f.write('start_time = \''+start_time+'\'\n')
    f.write('duration_s = \''+str(nproc)+' days\'\n')

    f.close()

slurm_e2e = coderoot+"slurm_e2e_sim"+isim+".sl"

slurm = """#!/bin/bash -l
#SBATCH -N 1 #COMPLETE HERE
#SBATCH -n {nproc}
#SBATCH -c 2 #COMPLETE HERE
#SBATCH -C haswell
#SBATCH -q debug #regular
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00 #COMPLETE HERE #e.g. 00:30:00
#SBATCH -o sim{isim}_{det_names_file}.log

cd {coderoot}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=2

srun python -c "from e2e_simulation import e2e_sim_production;

e2e_sim_production('{toml_filename}')"

"""

slurm = slurm.format(**locals())
f = open(slurm_e2e, "wt")
f.write(slurm)
f.close()

process = subprocess.Popen("sbatch "+slurm_e2e, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(stdout_data, stderr_data) = process.communicate()

print(stdout_data)
print(stderr_data)
