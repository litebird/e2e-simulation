import numpy as np
import subprocess
import sys

#command line example (collecting jobs' id in job_id.txt file):
#python run_simulation.py 0 LFT L2-050 >> job_id.txt

#parameters
isim            = sys.argv[1].zfill(4)
telescope       = sys.argv[2] #e.g. 'LFT'
channel         = sys.argv[3] #e.g. 'L2-050'
det_names_file  = 'detectors_'+telescope+'_'+channel+'_T+B'
nside           = 512
start_time      = '2030-04-01T00:00:00'
ntasks_per_node = 48
sim_days        = 365 #simulated days
mapmaking_type  = 'binned' #binned, destriper or all
imo_version     = 'v1.3'
name            = 'sim'+isim+'_'+det_names_file

#empirical values for nodes and time needed for sims > 0000
match = channel[0:2]
if match == "L1" :
    nnodese2e   = 4
    nnodesmadam = 3
    walle2e   = "00:30:00"
    wallmadam = "00:30:00"
if match == "L2" :
    nnodese2e   = 2
    nnodesmadam = 2
    walle2e   = "00:15:00"
    wallmadam = "00:15:00"
if match == "L3" :
    nnodese2e   = 4
    nnodesmadam = 3
    walle2e   = "00:30:00"
    wallmadam = "00:30:00"
if match == "L4" :
    nnodese2e   = 4
    nnodesmadam = 3
    walle2e   = "00:30:00"
    wallmadam = "00:30:00"
if match == "H1" :
    nnodese2e   = 7
    nnodesmadam = 4
    walle2e   = "00:20:00"
    wallmadam = "00:30:00"
if match == "H2" :
    nnodese2e   = 7
    nnodesmadam = 4
    walle2e   = "00:20:00"
    wallmadam = "00:30:00"
if match == "H3" :
    nnodese2e   = 10
    nnodesmadam = 6
    walle2e   = "00:20:00"
    wallmadam = "00:40:00"
if match == "M1" :
    nnodese2e   = 10
    nnodesmadam = 10
    walle2e   = "00:20:00"
    wallmadam = "01:00:00"
if match == "M2" :
    nnodese2e   = 14
    nnodesmadam = 9
    walle2e   = "00:30:00"
    wallmadam = "01:00:00"

partition       = '#SBATCH --partition=skl_usr_prod                 #The name of queue to use' if nnodese2e>2 else '#SBATCH --partition=skl_usr_dbg                  #The name of queue to use'
qos_bprod       = '#SBATCH --qos=skl_qos_bprod                      #for 54 nodes' if nnodese2e>32 else ''

#paths
coderoot        = '' #COMPLETE HERE   #folder where e2e_simulation.py is stored
base_path       = '/my/path/litebird/e2e_ns'+str(nside)+'/sim'+isim+'/'+det_names_file+'/' #COMPLETE HERE   #folder where you want to save the output files
input_maps_path = '/global/cfs/cdirs/litebird/simulations/maps/post_ptep_inputs_20220522/beam_convolved/'
madam_path      = '/my/path/Madam3.7.4/' #folder where madam executable is stored
user_email      = '' #COMPLETE HERE   #your email for notification


#create TOML file for e2e_simulation.py
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



#run e2e_simulation.py
slurm_e2e = coderoot+"slurm_e2e_sim"+isim+"_"+det_names_file+".sl"

slurm = """#!/bin/bash
#SBATCH --time={walle2e}                          #The requested execution time (max time) in hh:mm:ss
#SBATCH --nodes={nnodese2e}                         #The number of requested nodes
#SBATCH --ntasks-per-node={ntasks_per_node}      #The number of requested tasks/node
#SBATCH --cpus-per-task=1
#SBATCH --mem=182000                             #The requested memory per node
#SBATCH --job-name e2e_simulation                #The job name
#SBATCH --account=INF23_litebird                 #Project name
{partition}
{qos_bprod}
#SBATCH --mail-type=ALL                          #Send me an email at job start/end
#SBATCH --mail-user={user_email}                 #User mail address
#SBATCH --output=sim{isim}_{det_names_file}.out
#SBATCH --error=sim{isim}_{det_names_file}.err

cd {coderoot}
#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads
#export OMP_NUM_THREADS=1

srun --cpu-bind=cores python -c "from e2e_simulation import e2e_sim_production;
e2e_sim_production('{toml_filename}')"
"""

slurm = slurm.format(**locals())
f = open(slurm_e2e, "wt")
f.write(slurm)
f.close()

process = subprocess.Popen("sbatch "+slurm_e2e, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(stdout_data, stderr_data) = process.communicate()



#print useful information
print("sim"+isim,det_names_file+'\n')

print("e2e")
print("out: "+str(stdout_data).split('b\'')[1][:-3])
print("err: "+str(stderr_data).split('b\'')[1][:-3])
print('')



#run madam
if(mapmaking_type=='all' or mapmaking_type=='destriper'):

    #get e2e job id
    slurm_e2e_job_id = str(int(stdout_data[-9:]))

    link = 'false' #useful for producing links to tods and pointings for saving memory only for cases different than cmb_fg_wn_1f_100mHz

    if(int(isim)==0):
        madam_maps_list = ['cmb_fg_wn_1f_100mHz',
                           'cmb_fg_wn_1f_30mHz',
                           'wn_1f_100mHz',
                           'wn_1f_30mHz']
    else:
        madam_maps_list = ['cmb_fg_wn_1f_100mHz',
                           'cmb_fg_wn_1f_30mHz']

    for madam_map in madam_maps_list:
        slurm_madam = coderoot+"slurm_madam_"+madam_map+"_sim"+isim+"_"+det_names_file+".sl"
        
        slurm = """#!/bin/bash
#SBATCH --time={wallmadam}                          #The requested execution time (max time) in hh:mm:ss
#SBATCH --nodes={nnodesmadam}                         #The number of requested nodes
#SBATCH --ntasks-per-node={ntasks_per_node}      #The number of requested tasks/node
#SBATCH --cpus-per-task=1
#SBATCH --mem=182000                             #The requested memory per node
#SBATCH --job-name madam_{madam_map}             #The job name
#SBATCH --account=INF23_litebird                 #Project name
{partition}
{qos_bprod}
#SBATCH --mail-type=ALL                          #Send me an email at job start/end
#SBATCH --mail-user={user_email}                 #User mail address
#SBATCH --output=sim{isim}_madam_{madam_map}_{det_names_file}.out
#SBATCH --error=sim{isim}_madam_{madam_map}_{det_names_file}.err

cd {base_path}
if {link}
then
  ln -sr madam_cmb_fg_wn_1f_100mHz/*.fits madam_{madam_map}/
fi

cd {madam_path}
srun --cpu-bind=cores ./madam {base_path}madam_{madam_map}/madam.par
"""
        slurm = slurm.format(**locals())
        f = open(slurm_madam, "wt")
        f.write(slurm)
        f.close()

        link = 'true'

        process = subprocess.Popen("sbatch -d afterok:"+slurm_e2e_job_id+" "+slurm_madam, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout_data, stderr_data) = process.communicate()

        #print useful information
        print("madam "+madam_map)
        print("out: "+str(stdout_data).split('b\'')[1][:-3])
        print("err: "+str(stderr_data).split('b\'')[1][:-3])
        print('')

#separate chunks of output for different channels
print('-'*42)
print('')
