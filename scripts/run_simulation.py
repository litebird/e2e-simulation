import numpy as np
import subprocess
import sys

# command line example (collecting jobs' id in job_id.txt file):
# python run_simulation.py 0 1 LFT L2-050 >> job_id.txt

# parameters
# general
isimstart = sys.argv[1].zfill(4)  # from which simulation to start
isimend = sys.argv[2].zfill(4)  # last sim
channel = sys.argv[3]  # e.g. 'LF1_40'
telescope = "LMHFT" 
#Detectors: three possibilities
#A file with a list of detectors to use
#The string "all" for using all the detectors in the IMo
#Integer n for using the first n detectors in the IMo
detectors = "all"
ntasks_per_node = 48
mapmaking_type = "binned"  # brahmap or all or False
# simulation
start_time = "2034-04-01T00:00:00" # either a ``float`` or a ``astropy.time.Time``
sim_days = 365  # simulated days
want_CMB = True
nside = 512
lmax = 3*nside-1
mmax = 4
CMB_seed = 1234
want_FG = True
FG_model = "low_complexity"  # high_complexity
want_signal_per_detector = False # if false generates the same sky for all the detectors
use_hwp = False
want_BP_integration = False
want_dipole_signal = False
want_2f = False
want_non_linearity = False
want_gain_drift = False
tod_method = "scan"  # convolution
noise = "white"  # one_over_f or False
save_tod = False
save_invcovpp = False
imo_location = "/my/path/litebird/IMo_LiteBIRD/Reformation_Plan/option1M/"  # location of the file schema.json
imo_version = "IMo_vReformationPlan_Option1M"

name = "sim_from" + isimstart + "to" + isimend + "_" + channel + "_" +str(detectors)

match = channel[0:3]
if match == "MF1":
    nnodese2e = 6   
    walle2e = "06:00:00"

partition = (
    "#SBATCH --partition=g100_usr_prod                 #The name of queue to use"
    if nnodese2e > 2
    else "#SBATCH --partition=g100_usr_dbg                  #The name of queue to use"
)

# paths
coderoot = ""  # COMPLETE HERE   #folder where e2e_simulation.py is stored
base_path_prefix = (
    "/my/path/litebird/e2e_ns" + str(nside) + "/"
)  # COMPLETE HERE   #folder where you want to save the output files; sim and channel info added later
base_path = coderoot + base_path_prefix
input_maps_path = "/global/cfs/cdirs/litebird/simulations/maps/post_ptep_inputs_20220522/beam_convolved/"
user_email = ""  # COMPLETE HERE   #your email for notification

# create TOML files for e2e_simulation.py for each isim
toml_filename = (
    "e2e_" + name + "_params"
)
with open(coderoot + "../ancillary/" + toml_filename + ".toml", "w") as f:
    f.write("[general]\n")
    f.write("imo_location = '" + imo_location + "'\n")
    f.write("imo_version = '" + imo_version + "'\n")
    f.write("input_maps_path = '" + input_maps_path + "'\n")
    f.write("telescope = '" + telescope + "'\n")
    f.write("channel = '" + channel + "'\n")
    f.write("detectors = '" + str(detectors) + "'\n")
    f.write("mission_time_days = '" + str(sim_days) + "'\n")
    f.write("mapmaking_type = '" + mapmaking_type + "'\n")
    f.write("[simulation]\n")
    f.write("name = '" + name + "'\n")
    f.write("base_path = '" + base_path + "'\n")
    f.write("start_time = '" + start_time + "'\n")
    f.write("duration_s = '" + str(sim_days) + " days'\n")
    f.write("nside = " + str(nside) + "\n")
    f.write("lmax = " + str(lmax) + "\n")
    f.write("mmax = " + str(mmax) + "\n")
    f.write("want_CMB = '" + str(want_CMB) + "'\n")
    f.write("CMB_seed = " + str(CMB_seed) + "\n")
    f.write("want_FG = '" + str(want_FG) + "'\n")
    f.write("FG_model = '" + str(FG_model) + "'\n")
    f.write("want_signal_per_detector = '" + str(want_signal_per_detector) + "'\n")
    f.write("want_BP_integration = '" + str(want_BP_integration) + "'\n")
    f.write("want_dipole_signal = '" + str(want_dipole_signal) + "'\n")
    f.write("want_2f = '" + str(want_2f) + "'\n")
    f.write("want_non_linearity = '" + str(want_non_linearity) + "'\n")
    f.write("want_gain_drift = '" + str(want_gain_drift) + "'\n")
    f.write("tod_method = '" + tod_method + "'\n")
    f.write("use_hwp = '" + str(use_hwp) + "'\n")
    f.write("noise = '" + noise + "'\n")
    f.write("save_tod = '" + str(save_tod) + "'\n")
    f.write("save_invcovpp = '" + str(save_invcovpp) + "'\n")

    f.close()


# run e2e_simulation.py
slurm_e2e = coderoot + "slurm_" + name + ".sl" 

slurm = """#!/bin/bash
#SBATCH --time={walle2e}                         #The requested execution time (max time) in hh:mm:ss
#SBATCH --nodes={nnodese2e}                      #The number of requested nodes
#SBATCH --ntasks-per-node={ntasks_per_node}      #The number of requested tasks/node
#SBATCH --cpus-per-task=1
#SBATCH --mem=375300                             #The requested memory per node
#SBATCH --job-name={name}                        #The job name
#SBATCH --account=INF25_litebird_1               #Project name
{partition}
#SBATCH --mail-type=ALL                          #Send me an email at job start/end
#SBATCH --mail-user={user_email}                 #User mail address
#SBATCH --output={name}.out
#SBATCH --error={name}.err

cd {coderoot}
export OMP_NUM_THREADS=1

srun python -c "from e2e_simulation import e2e_sim_production;
e2e_sim_production('{toml_filename}','{isimstart}','{isimend}')"
"""

slurm = slurm.format(**locals())
f = open(slurm_e2e, "wt")
f.write(slurm)
f.close()

process = subprocess.Popen(
    "sbatch " + slurm_e2e, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
(stdout_data, stderr_data) = process.communicate()


# print useful information
print(str(detectors) + "_sim_from" + isimstart + "to" + isimend + "\n")

print("e2e")
print("out: " + str(stdout_data).split("b'")[1][:-3])
print("err: " + str(stderr_data).split("b'")[1][:-3])
print("")
