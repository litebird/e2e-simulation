import subprocess
import sys

import numpy as np

# command line example (collecting jobs' id in job_id.txt file):
# python run_simulation.py 0 LF1_40 1234 >> job_id.txt

# parameters
# general
isim = sys.argv[1].zfill(4)  # index of the simulation
channel = sys.argv[2] # e.g. "LF1_40"
option = sys.argv[3] # e.g. "Option1"

simulation_seed = isim
telescope = "LMHFT"
save_det_folder = f"/dss/dssfs02/lwp-dss-0001/pn36hu/pn36hu-dss-0000/beam_sims/det_files/{option}/"

# Detectors: three possibilities
# A file with a list of detectors to use
# The string "all" for using all the detectors in the IMo
# Integer n for using the first n detectors in the IMo
detectors = save_det_folder + "detectors_" + telescope + "_" + channel + ".txt"
ntasks_per_node = 96
# simulation
start_time = "2034-04-01T00:00:00"  # ``astropy.time.Time``
sim_days = 365  # simulated days
want_CMB = False ###FIXME
CMB_seed = 5678
nside = 512
lmax = 2 * nside
mmax = lmax - 4
want_FG = True
FG_model = "low_complexity"  # medium_complexity, high_complexity
want_signal_per_detector = False  # if false generates the same sky for all the detectors
want_BP_integration = False
want_dipole_signal = False
want_2f = False
want_non_linearity = False
want_gain_drift = False
tod_method = "convolution"  # scan
noise = "white" #or one_over_f
save_tod = False
save_invcovpp = False

if option == "Option1":
    use_hwp = False
    imo_location = "/dss/dssfs02/lwp-dss-0001/pn36hu/pn36hu-dss-0000/beam_sims/IMo_vPostKDP2/json1/"  # location of schema.json
    imo_version = "IMo_vPostKDP2_Option1"
elif option == "Option2":
    use_hwp = True
    imo_location = "/dss/dssfs02/lwp-dss-0001/pn36hu/pn36hu-dss-0000/beam_sims/IMo_vPostKDP2/json2/"  # location of schema.json
    imo_version = "IMo_vPostKDP2_Option2"

mapmaking_type = "binned"  # brahmap or all or False

if detectors != "all" and not isinstance(detectors, int):
    name = "sim_" + str(isim).zfill(4) + "_" + channel + "_custom_det"
else:
    name = "sim_" + str(isim).zfill(4) + "_" + channel + "_" + str(detectors)

nnodese2e = 2
walle2e = "00:30:00"

clusters = "#SBATCH --clusters=cm4"
partition = "#SBATCH --partition=cm4_std"
qos = "#SBATCH --qos=cm4_std"

# paths
coderoot = "/dss/dssfs02/lwp-dss-0001/pn36hu/pn36hu-dss-0000/beam_sims/e2e-simulation/scripts/"  # COMPLETE HERE   #folder where e2e_simulation.py is stored
base_path = (
    "/dss/dssfs02/lwp-dss-0001/pn36hu/pn36hu-dss-0000/beam_sims/output/e2e_SL_ns" + str(nside) + "/"
)  # COMPLETE HERE   #folder where you want to save the output files; sim and channel info added later
user_email = "marta.monelli@ipmu.jp"  # COMPLETE HERE   #your email for notification

# create TOML files for e2e_simulation.py for each isim
toml_filename = "/dss/dssfs02/lwp-dss-0001/pn36hu/pn36hu-dss-0000/beam_sims/output/params/" + "e2e_" + name + "_params" + ".toml"

with open(toml_filename, "w") as f:
    f.write("[general]\n")
    f.write("imo_location = '" + imo_location + "'\n")
    f.write("imo_version = '" + imo_version + "'\n")
    f.write("telescope = '" + telescope + "'\n")
    f.write("detectors = '" + str(detectors) + "'\n")
    f.write("mission_time_days = '" + str(sim_days) + "'\n")
    f.write("[simulation]\n")
    f.write("name = '" + name + "'\n")
    f.write("base_path = '" + base_path + "'\n")
    f.write("start_time = '" + start_time + "'\n")
    f.write("duration_s = '" + str(sim_days*24*60*60) + " seconds'\n")
    f.write("nside = " + str(nside) + "\n")
    f.write("lmax = " + str(lmax) + "\n")
    f.write("mmax = " + str(mmax) + "\n")
    f.write("want_CMB = "+ str(want_CMB).lower() +"\n")
    f.write("CMB_seed = " + str(CMB_seed) + "\n")
    f.write("want_FG = " + str(want_FG).lower()+ "\n")
    f.write("FG_model = '" + FG_model + "'\n")
    f.write("want_signal_per_detector = " + str(want_signal_per_detector).lower() + "\n")
    f.write("want_BP_integration = " + str(want_BP_integration).lower() + "\n")
    f.write("want_dipole_signal = " + str(want_dipole_signal).lower() + "\n")
    f.write("want_2f = " + str(want_2f).lower() + "\n")
    f.write("want_non_linearity = " + str(want_non_linearity).lower() + "\n")
    f.write("want_gain_drift = " + str(want_gain_drift).lower() + "\n")
    f.write("tod_method = '" + tod_method + "'\n")
    f.write("use_hwp = " + str(use_hwp).lower() + "\n")
    f.write("noise = '" + noise + "'\n")
    f.write("save_tod = " + str(save_tod).lower() + "\n")
    f.write("save_invcovpp = " + str(save_invcovpp).lower() + "\n")
    f.write("mapmaking_type = '" + mapmaking_type + "'\n")

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
{clusters}
{partition}
{qos}
#SBATCH --mail-type=ALL                          #Send me an email at job start/end
#SBATCH --mail-user={user_email}                 #User mail address
#SBATCH --output={name}.out
#SBATCH --error={name}.err

cd {coderoot}
export OMP_NUM_THREADS=1

srun python -c "from e2e_simulation_SL import e2e_sim_production;
e2e_sim_production('{toml_filename}','{isim}','{channel}','{simulation_seed}')"
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
print(str(detectors) + "_sim_" + isim + "\n")

print("e2e")
print("out: " + str(stdout_data).split("b'")[1][:-3])
print("err: " + str(stderr_data).split("b'")[1][:-3])
print("")
