import numpy as np
import subprocess

#paths
coderoot   = #COMPLETE HERE   #folder where e2e_simulation.py is stored
base_path  = #COMPLETE HERE   #folder where you want to save the output files
user_email = #COMPLETE HERE   #your email for notification

#parameters
telescope      = 'LFT'
det_names_file = 'detectors_LFT_L2-050_T+B'#.txt
nside          = 512
isimstart      = 0
nsims          = 2
nproc          = 30 #number of simulation days, too
mapmaking_type = 'destriper' #binned, destriper or all



#loop over simulations
for isim in np.arange(isimstart,nsims,1):
    print("isim:",isim)

    isim_str = str(isim).zfill(2)

    slurm_e2e = coderoot+"slurm_e2e_sim"+isim_str+".sl"

    slurm = """#!/bin/bash -l
#SBATCH -N #COMPLETE HERE
#SBATCH -n {nproc}
#SBATCH -c #COMPLETE HERE
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=ALL
#SBATCH -t #COMPLETE HERE #e.g. 00:30:00
#SBATCH -o sim{isim_str}.log

cd {coderoot}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=1

srun python -c "from e2e_simulation import e2e_sim_production;

e2e_sim_production(telescope='{telescope}',
                   det_names_file='{det_names_file}',
                   nside={nside},
                   mission_time_days={nproc},
                   isim={isim},
                   base_path='{base_path}',
                   mapmaking_type='{mapmaking_type}')"
"""
    slurm = slurm.format(**locals())
    f = open(slurm_e2e, "wt")
    f.write(slurm)
    f.close()

    process = subprocess.Popen("sbatch "+slurm_e2e, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout_data, stderr_data) = process.communicate()

    print(stdout_data)
    print(stderr_data)
    print()
