import numpy as np
import subprocess
import sys

#command line example (collecting jobs' id in job_id.txt file):
#python run_simulation.py 0 1 LFT L2-050 >> job_id.txt

#parameters
isimstart       = sys.argv[1].zfill(4) #from which simulation to start
isimend         = sys.argv[2].zfill(4) #last sim
telescope       = sys.argv[3] #e.g. 'LFT'
channel         = sys.argv[4] #e.g. 'L2-050'
det_names_file  = 'detectors_'+telescope+'_'+channel+'_T+B'
nside           = 512
start_time      = '2030-04-01T00:00:00'
ntasks_per_node = 48
sim_days        = 365 #simulated days
mapmaking_type  = 'binned' #binned, destriper or all
imo_location    = '/my/path/litebird/IMo_LiteBIRD/Reformation_Plan/option1M/' #location of the file schema.json 
imo_version     = 'v1.3'
name            = 'sim_from'+isimstart+'to'+isimend+'_'+det_names_file

#empirical values for nodes and time needed for sims > 0000
match = channel[0:2]
if match == 'L1' :
    nnodese2e   = 6
    walle2e   = '06:00:00'
if match == 'L2' :
    nnodese2e   = 4
    walle2e   = '06:00:00'
if match == 'L3' :
    nnodese2e   = 6
    walle2e   = '06:00:00'
if match == 'L4' :
    nnodese2e   = 6
    walle2e   = '06:00:00'
if match == 'H1' :
    nnodese2e   = 10
    walle2e   = '06:00:00'
if match == 'H2' :
    nnodese2e   = 10
    walle2e   = '06:00:00'
if match == 'H3' :
    nnodese2e   = 13
    walle2e   = '06:00:00'
if match == 'M1' :
    nnodese2e   = 14
    walle2e   = '06:00:00'
if match == 'M2' :
    nnodese2e   = 19
    walle2e   = '06:00:00'

partition       = '#SBATCH --partition=skl_usr_prod                 #The name of queue to use' if nnodese2e>2 else '#SBATCH --partition=skl_usr_dbg                  #The name of queue to use'

#paths
coderoot         = '' #COMPLETE HERE   #folder where e2e_simulation.py is stored
base_path_prefix = '/my/path/litebird/e2e_ns'+str(nside)+'/' #COMPLETE HERE   #folder where you want to save the output files; sim and channel info added later
input_maps_path  = '/global/cfs/cdirs/litebird/simulations/maps/post_ptep_inputs_20220522/beam_convolved/'
user_email       = '' #COMPLETE HERE   #your email for notification

#create TOML files for e2e_simulation.py for each isim
toml_filename = 'e2e_sim_from'+isimstart+'to'+isimend+'_'+det_names_file+'_params'
with open(coderoot+'../ancillary/'+toml_filename+'.toml', 'w') as f:
    f.write('[general]\n')
    f.write('imo_location = \''+imo_location+'\'\n')
    f.write('imo_version = \''+imo_version+'\'\n')
    f.write('input_maps_path = \''+input_maps_path+'\'\n')
    f.write('telescope = \''+telescope+'\'\n')
    f.write('det_names_file = \''+det_names_file+'\'\n')
    f.write('nside = '+str(nside)+'\n')
    f.write('mission_time_days = \''+str(sim_days)+'\'\n')
    f.write('mapmaking_type = \''+mapmaking_type+'\'\n')
    f.write('[simulation]\n')
    f.write('name = \''+name+'\'\n')
    f.write('base_path = \''+base_path+'\'\n')
    f.write('start_time = \''+start_time+'\'\n')
    f.write('duration_s = \''+str(sim_days)+' days\'\n')

    f.close()



#run e2e_simulation.py
slurm_e2e = coderoot+'slurm_e2e_sim_from'+isimstart+'to'+isimend+'_'+det_names_file+'.sl'

slurm = '''#!/bin/bash
#SBATCH --time={walle2e}                          #The requested execution time (max time) in hh:mm:ss
#SBATCH --nodes={nnodese2e}                         #The number of requested nodes
#SBATCH --ntasks-per-node={ntasks_per_node}      #The number of requested tasks/node
#SBATCH --cpus-per-task=1
#SBATCH --mem=375300                             #The requested memory per node
#SBATCH --job-name e2e_{det_names_file}_sim_from{isimstart}to{isimend}                #The job name
#SBATCH --account=INF25_litebird_1               #Project name
{partition}
#SBATCH --mail-type=ALL                          #Send me an email at job start/end
#SBATCH --mail-user={user_email}                 #User mail address
#SBATCH --output={det_names_file}_sim_from{isimstart}to{isimend}.out
#SBATCH --error={det_names_file}_sim_from{isimstart}to{isimend}.err

cd {coderoot}
#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads
#export OMP_NUM_THREADS=1

srun --cpu-bind=cores python -c "from e2e_simulation import e2e_sim_production;
e2e_sim_production('{toml_filename}','{isimstart}','{isimend}')"
'''

slurm = slurm.format(**locals())
f = open(slurm_e2e, 'wt')
f.write(slurm)
f.close()

process = subprocess.Popen('sbatch '+slurm_e2e, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(stdout_data, stderr_data) = process.communicate()


#print useful information
print(det_names_file+'_sim_from'+isimstart+'to'+isimend+'\n')

print('e2e')
print('out: '+str(stdout_data).split('b\'')[1][:-3])
print('err: '+str(stderr_data).split('b\'')[1][:-3])
print('')
