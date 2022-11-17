import numpy as np
import subprocess
import sys

#command line example:
#python run_simulation.py 0 LFT L2-050

#parameters
isim            = sys.argv[1].zfill(4)
telescope       = sys.argv[2] #e.g. 'LFT'
channel         = sys.argv[3] #e.g. 'L2-050'
det_names_file  = 'detectors_'+telescope+'_'+channel+'_T+B'
nside           = 512
start_time      = '2030-04-01T00:00:00'

nnodes          = 27 #54 for MFT (largest number of detectors)
ntasks_per_node = 48
sim_days        = 365 #simulated days

mapmaking_type  = 'destriper' #binned, destriper or all
imo_version     = 'v1.3'
name            = 'sim'+isim+'_'+det_names_file

#paths
coderoot        = '' #COMPLETE HERE   #folder where e2e_simulation.py is stored
base_path       = '/my/path/litebird/e2e_ns'+str(nside)+'/sim'+isim+'/'+det_names_file+'/' #COMPLETE HERE   #folder where you want to save the output files
input_maps_path = '/global/cfs/cdirs/litebird/simulations/maps/post_ptep_inputs_20220522/beam_convolved/'
user_email      = '' #COMPLETE HERE   #your email for notification




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
    f.write('mission_time_days = \''+str(sim_days)+'\'\n')
    f.write('mapmaking_type = \''+mapmaking_type+'\'\n')
    f.write('[simulation]\n')
    f.write('name = \''+name+'\'\n')
    f.write('base_path = \''+base_path+'\'\n')
    f.write('start_time = \''+start_time+'\'\n')
    f.write('duration_s = \''+str(sim_days)+' days\'\n')

    f.close()

slurm_e2e = coderoot+"slurm_e2e_sim"+isim+"_"+det_names_file+".sl"

slurm = """#!/bin/bash
#SBATCH --time=00:30:00                          #The requested execution time (max time) in hh:mm:ss
#SBATCH --nodes={nnodes}                         #The number of requested nodes
#SBATCH --ntasks-per-node={ntasks_per_node}      #The number of requested tasks/node
#SBATCH --cpus-per-task=1
#SBATCH --mem=182000                             #The requested memory per node
#SBATCH --job-name e2e_simulation                #The job name
#SBATCH --account=INF22_lspe                     #Project name
#SBATCH --partition=skl_usr_prod                 #The name of queue to use #SBATCH --qos=skl_qos_bprod for 54 nodes
#SBATCH --mail-type=ALL                          #Send me an email at job start/end
#SBATCH --mail-user={user_email}                 #User mail address
#SBATCH --output=sim{isim}_{det_names_file}.out
#SBATCH --error=sim{isim}_{det_names_file}.err

cd {coderoot}
#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads
#export OMP_NUM_THREADS=2

srun --cpu-bind=cores python -c "from e2e_simulation import e2e_sim_production;

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
