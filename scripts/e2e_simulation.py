import litebird_sim as lbs
import numpy as np
import healpy as hp
import matplotlib.pylab as plt
from astropy.time import Time
import time
from typing import Union
from pathlib import Path
import os
import sys


def e2e_sim_production(det_names_file,
    isimstart: int,
    isimend: Union[None, int] = None,
    ):
    '''
    This function reads CMB/FG maps, scans them and produces white noise, 1/f noises and cmb dipole.
    Only for the first simulation, i.e. sim0000, it writes timelines (cmb, fg, white noise,
    white noise+1/f noise with f_knee of 30mHz, white noise+1/f noise with f_knee of 100mHz, dipole)
    as hdf5 files. It then uses the timelines to produce binned and/or to save the results to then
    produce destriped maps with madam.
    The time employed for each step is printed.

    toml_filename: string, name of the TOML file where the following parameters are specified:
        imo_version: string, version of the IMO, e.g. 'v1.3';
        input_maps_path: string, location of the input maps to be scanned;
        telescope: string, telescope name, e.g. 'LFT';
        det_names_file: string, file containing detector names, each one associated with its channel and noise NET.
                        Only and all detectors in this file will be used, e.g. to consider only top
                        detectors you should produce a file only with those detectors. This string is
                        also used for names of saved files.
                        IMPORTANT: this script should be run with only 1 channel;
        nside: int, the resolution of the maps;
        isim: int, simulation number;
        mission_time_days: string, days of observation;
        mapmaking_type: string, type of mapmaking, 'binned', 'destriper' or 'all';
        name: string, name of the simulation;
        base_path: string, path where you want to save the maps and observations generated;
        start_time: string, start time of the simulation, e.g. '2030-04-01T00:00:00';
        duration_s: string, days of observation, e.g. '365 days' (same as mission_time_days but recognized by lbs.Simulation)
    '''

    #for parallelization
    comm = lbs.MPI_COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if(rank==0):
        t_in = time.time()

    #initializing the IMO
    imo = lbs.Imo()

    if isimend == None:
        isimend = isimstart

    #loop over simulations
    first_time = True

    for isim in range(isimstart,isimend+1):
        if(first_time):
            #initializing the simulation
            sim = lbs.Simulation(parameter_file=os.path.dirname(os.getcwd())+'/ancillary/e2e_sim'+str(isim).zfill(4)+'_'+det_names_file+'_params.toml',
                                 mpi_comm=comm)

            #extract useful parameters
            imo_version       =     sim.parameters['general']['imo_version']
            input_maps_path   =     sim.parameters['general']['input_maps_path']
            telescope         =     sim.parameters['general']['telescope']
            det_names_file    =     sim.parameters['general']['det_names_file']
            nside             = int(sim.parameters['general']['nside'])
            mission_time_days =     sim.parameters['general']['mission_time_days']

            base_path         =     sim.parameters['simulation']['base_path']
            duration_s        =     sim.parameters['simulation']['duration_s']
            start_time        =     sim.parameters['simulation']['start_time']

            #create new base path folder
            if(rank==0):
                if not os.path.exists(base_path):
                    os.makedirs(base_path)

            #create save path for output maps
            map_path = base_path+'maps/'
            if(rank==0):
                if not os.path.exists(map_path):
                    os.mkdir(map_path)
            
            #read channel, noise and detector names
            det_names_file_path = os.path.dirname(os.getcwd())+'/ancillary/'+det_names_file+'.txt'
            det_file = np.genfromtxt(det_names_file_path,
                                     skip_header=1,
                                     dtype=str)

            channels = det_file[:,1]
            noises   = det_file[:,4].astype(dtype=float)
            detnames = det_file[:,5]

            #get frequency (IMPORTANT: this script should be run with only 1 channel)
            freq = int(channels[0][3:6]) #e.g.: channels[0] = 'L2-050' --> freq = 50

            #number of detectors = raws of {det_names_file}.txt
            ndet = np.size(detnames)

            #loading the instrument metadata
            inst_info = sim.imo.query('/releases/'+imo_version+'/satellite/'+telescope+'/instrument_info')

            #generating the quaternions of the instrument
            sim.generate_spin2ecl_quaternions(imo_url='/releases/'+imo_version+'/satellite/scanning_parameters/')

            #loading instrument info
            inst = lbs.InstrumentInfo(name=telescope, 
                                      boresight_rotangle_rad=np.deg2rad(inst_info.metadata['boresight_rotangle_deg']),
                                      spin_boresight_angle_rad=np.deg2rad(inst_info.metadata['spin_boresight_angle_deg']),
                                      spin_rotangle_rad=np.deg2rad(inst_info.metadata['spin_rotangle_deg']))
    
            #filling dets with info and detquats with quaternions of the detectors in detlist
            dets = []
            for i_det in range(ndet):
                det = lbs.DetectorInfo.from_imo(url='/releases/'+imo_version+'/satellite/'+telescope+'/'+channels[i_det]+'/'+detnames[i_det]+'/detector_info',
                                                imo=imo)
                dets.append(det)

            if(rank==0):
                t_sim = time.time()
                print('simulation time: ',t_sim-t_in)

            comm.barrier()

            #create Observation object
            (obs_multitod,) = sim.create_observations(detectors=dets,
                                                      n_blocks_det=1,
                                                      n_blocks_time=size,
                                                      split_list_over_processes=False)

            obs_multitod.tod_cmb_fg_wn_1f_100mHz = np.zeros_like(obs_multitod.tod) 
            obs_multitod.tod                     = np.array([], dtype='float32') #not used, to save memory
            obs_multitod.tod_cmb_fg_wn_1f_30mHz  = np.zeros_like(obs_multitod.tod_cmb_fg_wn_1f_100mHz)

            comm.barrier()

            #hwp specification
            hwp_radpsec = inst_info.metadata['hwp_rpm']*2*np.pi/60

            #get pointings and store them in obs_multitod
            quaternion_buffer = np.zeros((obs_multitod.n_samples, 1, 4))
            pointings = lbs.pointings.get_pointings(obs_multitod,
                                                    spin2ecliptic_quats=sim.spin2ecliptic_quats,
                                                    bore2spin_quat=inst.bore2spin_quat,
                                                    hwp=lbs.IdealHWP(hwp_radpsec),   #applies hwp rotation angle to the polarization angle
                                                    quaternion_buffer=quaternion_buffer,
                                                    store_pointings_in_obs=True)     #if True, stores colatitude and longitude in obs_multitod.pointings,
                                                                                     #and the polarization angle in obs_multitod.psi
            del quaternion_buffer

            if(rank==0):
                t_point = time.time()
                print('time for pointings: ', t_point-t_sim)

        #end of first_time

        obs_multitod.tod_cmb_fg_wn_1f_100mHz = 0.0
        obs_multitod.tod_cmb_fg_wn_1f_30mHz = 0.0

        comm.barrier()

        if(rank==0):
            #load maps
            try:
                same_freq_spec = '' #specify which map to load for channels with the same frequency
                if freq in [68,78,89]:
                    if(channels[0][:2]=='L1' or channels[0][:2]=='L2'):
                        same_freq_spec = 'a'
                    else:
                        same_freq_spec = 'b'
                    #read cmb map only
                    maps =  hp.read_map(input_maps_path+'cmb/'+str(isim).zfill(2)+'/'+'LB_'+telescope+'_'+str(freq)+same_freq_spec+'_lens_cmb_postPTEP20220609.fits',
                                       field=[0,1,2])
                    #read and sum fg map to cmb one
                    maps += hp.read_map(input_maps_path+'all_fg/'                    +'LB_'+telescope+'_'+str(freq)+same_freq_spec+'_all_fg_postPTEP20220609.fits',
                                       field=[0,1,2])
            except:
                print('Error while reading map',input_map_type[i_m],'for channel',channels[0])
        else:
            maps = None

        #broadcast maps read by rank 0
        maps = comm.bcast(maps, root=0)
            
        #convert from uK to K #MBNR hard coded...
        maps *= 1e-6

        comm.barrier()

        #fill the TOD
        lbs.scan_map_in_observations(obs_multitod,
                                     maps,
                                     input_map_in_galactic=True,
                                     component='tod_cmb_fg_wn_1f_100mHz')

        obs_multitod.tod_cmb_fg_wn_1f_30mHz += obs_multitod.tod_cmb_fg_wn_1f_100mHz

        if(rank==0):
            t_scan = time.time()
            print('time for reading and scanning map for TODs: ', t_scan-t_point)

        comm.barrier()

        #pessimistic 1/f: set knee frequency and noise specification
        obs_multitod.fknee_mhz = 100
        obs_multitod.fmin_hz   = 1e-5
        obs_multitod.net_ukrts = noises

        #pessimistic 1/f: add noise
        lbs.add_noise_to_observations([obs_multitod],
                                      'one_over_f',
                                      scale=1,
                                      component='tod_cmb_fg_wn_1f_100mHz')

        #realistic 1/f: set knee frequency
        obs_multitod.fknee_mhz = 30

        #realistic 1/f: add noise
        lbs.add_noise_to_observations([obs_multitod],
                                      'one_over_f',
                                      scale=1,
                                      component='tod_cmb_fg_wn_1f_30mHz')

        if(rank==0):
            t_noise = time.time()
            print('time for filling noise timelines: ', t_noise-t_scan)

        comm.barrier()


        #create lists of combined components for mapmakers...
        obs_list_mapmaking = [['tod_cmb_fg_wn_1f_100mHz'],
                              ['tod_cmb_fg_wn_1f_30mHz']]
    
        #...and of their names
        obs_name_mapmaking = ['cmb_fg_wn_1f_100mHz',
                              'cmb_fg_wn_1f_30mHz']


        for obs_list,obs_name in zip(obs_list_mapmaking,obs_name_mapmaking):
            
            #build the output maps
            map_output = lbs.make_bin_map(obs_multitod,
                                          nside,
                                          do_covariance=False,
                                          output_map_in_galactic=True,
                                          components=obs_list)
            
            if(rank==0):
                map_name = 'LB_'+telescope+'_'+str(freq)+'_binned_'+obs_name+'_'+mission_time_days+'d'+'_'+str(isim).zfill(4)
                hp.write_map(map_path+map_name+'.fits',map_output,overwrite=True)

        comm.barrier()
    
        if(rank==0):
            t_save_maps = time.time()
            if(isim==0):
                print('time for mapmaking: ', t_save_maps-t_save_tod)
            else:
                print('time for mapmaking: ', t_save_maps-t_noise)


        first_time = False