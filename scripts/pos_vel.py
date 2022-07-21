import litebird_sim as lbs
import numpy as np
from astropy.time import Time
import os

#parameters (specify the parameters of your interest)
mission_time_days = 365
save_path         = './'
output_filename   = 'pos_vel_test.txt'

def pos_vel(mission_time_days,save_path,output_filename):
    '''
    This function calculates and saves the spacecraft positions and velocities.
    The output file, in txt format, will have 6 columns (x,y,z,vx,vy,vz) and mission_time_days+1 rows

    mission_time_days: days of observations;
    save_path: path where you want to save the file;
    output_filename: name of the output file (has to end with '.txt')
    '''

    #simulation start time
    start_time = Time('2030-04-01T00:00:00')

    orbit = lbs.SpacecraftOrbit(start_time)

    #spacecraft position and velocity
    pos_vel = lbs.spacecraft_pos_and_vel(orbit,
                                         start_time=start_time,
    									 time_span_s=86400.0*mission_time_days,
                                         delta_time_s=86400.0
                                         )

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    np.savetxt(save_path+output_filename,
               np.column_stack((pos_vel.positions_km,pos_vel.velocities_km_s)),
               header='pos_x \t\t  pos_y \t\t    pos_z \t\t      vel_x \t\t        vel_y \t\t          vel_z',
               )

pos_vel(mission_time_days,save_path,output_filename)