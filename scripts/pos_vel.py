import litebird_sim as lbs
import numpy as np
from astropy.time import Time
import os

#srun python -c "from pos_vel import pos_vel;pos_vel(toml_filename='e2e_sim000_detectors_LFT_L2-050_T+B_params')"

def pos_vel(toml_filename):
    '''
    This function calculates and saves the spacecraft positions and velocities.
    The output file, in txt format, will have 6 columns (x,y,z,vx,vy,vz) and mission_time_days+1 rows

    toml_filename: name of the TOML file where the following parameters are specified:
        base_path: path where file is saved;
        start_time: simulation start time;
        mission_time_days: days of observations.
    '''

    #initializing the simulation
    sim = lbs.Simulation(parameter_file=os.path.dirname(os.getcwd())+"/ancillary/"+toml_filename+".toml",
                         )

    #parameters and paths
    base_path         = sim.parameters["simulation"]["base_path"]
    start_time        = Time(sim.parameters["simulation"]["start_time"]) #simulation start time
    mission_time_days = sim.parameters["general"]["mission_time_days"]   #duration of the simulation

    orbit = lbs.SpacecraftOrbit(start_time)

    #spacecraft position and velocity
    pos_vel = lbs.spacecraft_pos_and_vel(orbit,
                                         start_time=start_time,
    									 time_span_s=86400.0*float(mission_time_days),
                                         delta_time_s=86400.0
                                         )

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    np.savetxt(base_path+'spacecraft_pos_vel_'+mission_time_days+'d.txt',
               np.column_stack((pos_vel.positions_km,pos_vel.velocities_km_s)),
               header='pos_x \t\t  pos_y \t\t    pos_z \t\t      vel_x \t\t        vel_y \t\t          vel_z',
               )

    print("Done")
