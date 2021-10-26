import litebird_sim as lbs
import numpy as np
import matplotlib.pylab as plt
import healpy as hp
import time
import astropy
from astropy.time import Time
import os
import pickle
import h5py


def call_to_fill_tod_maps(mission_time_days,T,nside,ni_det,nf_det, T_and_B):  
    '''
    Function parsing arguments to fill_tod_maps. 
    mission_time_days: days of observations;
    T: telescope name (string) e.g. 'LFT';
    ni_det: index of detector in the detectors list 'list_detectors_good_(L/M/H)FT' from which starting reading (integer number); 
    nf_det: index of detector in the detectors list 'list_detectors_good_(L/M/H)FT' up to which reading (integer number),
            i.e. detectors are read in the list from ni_det to nf_det; 
    T_and_B: True (if you want to include both Top&Bottom detectors between ni_det and nf_det) or False (if you want to include just Top detector)
    '''

    l_path = os.path.dirname(os.getcwd())+"/ancillary/list_detectors_good_"+T+".txt"

    l_file = np.genfromtxt(l_path, skip_header = 1, dtype = str)
    channel = l_file[ni_det:nf_det,1]
    noise = l_file[ni_det:nf_det,4].astype(dtype = float)
    detname_T = l_file[ni_det:nf_det,6]
    detname_B = l_file[ni_det:nf_det,7]
    
    fill_tod_maps(T, channel, detname_T, detname_B, noise, nside, mission_time_days, T_and_B)    

def fill_tod_maps(telescope, channel, detname_T, detname_B, noise, nside, mission_time_days, T_and_B):
    '''
    This function initializes a simulation, generates or reads a dictionaty of CMB/FG maps, one for each
    detector, and writes seven separated timelines (cmb,fg w/o band integration, fg w/ band integration,
    white noise, white+1/f noise, linear dipole, complete dipole) to be saved in separated hdf5 files. 
    The time employed for each step is printed.
    '''

    start_time = astropy.time.Time('2029-01-01T00:00:00')
    ndet = np.size(detname_T)
    if ndet == 1:
        base_path = "/tmp/sim_ns"+str(nside)+'_'+telescope+'_'+channel[0]
    else:
        base_path = "/tmp/sim_ns"+str(nside)+'_'+telescope+"_n_det_"+str(ndet)+'_'+str(mission_time_days)+'d'
    t_in = time.time()
    imo = lbs.Imo()

    sim = lbs.Simulation(base_path=base_path,#mpi_comm=comm,
                           start_time=start_time,duration_s=mission_time_days*24*3600.0)
    inst_info = sim.imo.query("/releases/v1.0/satellite/"+telescope+"/instrument_info")
    sim.generate_spin2ecl_quaternions(imo_url="/releases/v1.0/satellite/scanning_parameters/")
    inst = lbs.InstrumentInfo(name=telescope, 
        boresight_rotangle_rad=np.deg2rad(inst_info.metadata["boresight_rotangle_deg"]),
        spin_boresight_angle_rad=np.deg2rad(inst_info.metadata["spin_boresight_angle_deg"]),
        spin_rotangle_rad=np.deg2rad(inst_info.metadata["spin_rotangle_deg"]),)
    hwp_radpsec = inst_info.metadata["hwp_rpm"]*2*np.pi/60

    dets=[]
    detquats=[]
    for i in range(ndet):
        if T_and_B:
            for detname in (detname_T[i],detname_B[i]):
                det=lbs.DetectorInfo.from_imo(url="/releases/v1.0/satellite/"+telescope+"/"+channel[i]+"/"+detname+"/detector_info",imo=imo)
                dets.append(det)
                detquats.append(det.quat)
        else:
            det=lbs.DetectorInfo.from_imo(url="/releases/v1.0/satellite/"+telescope+"/"+channel[i]+"/"+detname_T[i]+"/detector_info",imo=imo)
            dets.append(det)
            detquats.append(det.quat)
    t_sim = time.time()
    print('simulation time: ',t_sim - t_in)

    #initialize several observations              
    (obs_cmb,) = sim.create_observations(detectors= dets,
        n_blocks_det = 1,
        n_blocks_time = 1,  #size,
        )

    (obs_fg,) = sim.create_observations(detectors= dets,
        n_blocks_det = 1,
        n_blocks_time = 1,  #size,
        )

    (obs_fg_bandint,) = sim.create_observations(detectors= dets,
        n_blocks_det = 1,
        n_blocks_time = 1,  #size,
        )

    (obs_noise_w,) = sim.create_observations(detectors= dets,
        n_blocks_det = 1,
        n_blocks_time = 1,  #size,
        )

    (obs_noise_w_1_f,) = sim.create_observations(detectors= dets,
        n_blocks_det = 1,
        n_blocks_time = 1,  #size,
        )
    
    t_obs = time.time()
    print('time for obs initialization: ', t_obs-t_sim)

    pointings = lbs.scanning.get_pointings(obs_cmb,
        spin2ecliptic_quats = sim.spin2ecliptic_quats,
        detector_quats = detquats,
        bore2spin_quat = inst.bore2spin_quat,)


    t_point = time.time()
    print('time for pointings: ', t_point-t_obs)

    M_cmb = [True,False,False]
    M_fg = [False,True,True]
    M_bandint = [False,False,True]
    obs = [obs_cmb,obs_fg,obs_fg_bandint]
    map_type = ['cmb','fg','fg_int']
    ch_info = []
    ch_names = ''
    for i in range(ndet):
        if channel[i] != channel[i-1] or i == 0:
            ch_info.append(lbs.FreqChannelInfo.from_imo(url="/releases/v1.0/satellite/"
                                           +telescope+"/"+channel[i]+"/channel_info",imo=imo))
            ch_names += '_'+channel[i]

    for i in range(3):
        map_path = base_path+'/'+map_type[i]+'_channels'+ch_names+'.pickle'
        print(map_path)
        if not os.path.isfile(map_path):
            Mbsparams = lbs.MbsParameters(
                make_cmb =M_cmb[i],
                make_fg = M_fg[i],
                seed_cmb = 1,
                fg_models = ["pysm_synch_0", "pysm_freefree_1","pysm_dust_0"],
                gaussian_smooth = True,
                bandpass_int = M_bandint[i],
                nside = nside,
                units = "K_CMB",
                maps_in_ecliptic = True,   #maps saved in ecliptic, because of dipole 
            )

            mbs = lbs.Mbs(simulation = sim,parameters = Mbsparams,channel_list = ch_info) #detector_list = dets)
            maps = mbs.run_all()[0]
            with open(map_path, 'wb') as f:
                pickle.dump(maps, f)
            t_map = time.time()
            if i == 0:
                print('time for creating map: ', t_map-t_point)
            else:
                print('time for creating map: ', t_map-t_tod)
        else:
            with open(map_path,'rb') as f:
                maps = pickle.load(f)
            t_map = time.time()
            if i == 0:
                print('time for reading map: ', t_map-t_point)
            else:
                print('time for reading map: ', t_map-t_tod)
        lbs.scan_map_in_observations(
                obs[i], pointings, hwp_radpsec, maps, fill_psi_and_pixind_in_obs=True
            )
        t_tod = time.time()
        print('time for writing tod: ', t_tod-t_map)


    noisetype = ['white','one_over_f']
    obs_noise = [obs_noise_w,obs_noise_w_1_f]
    for i_ob,ob in enumerate(obs_noise):
        print(noisetype[i_ob])
        ob.net_ukrts = []
        for i in range(ndet):
            ob.net_ukrts += [noise[i],noise[i]]
        lbs.add_noise([ob],noisetype[i_ob],scale = 1) #,random=random)
    t_noise = time.time()
    print('time for filling noise timeline: ', t_noise-t_tod)

    (obs_dipole_0,) = sim.create_observations(detectors= dets,
    n_blocks_det = 1,
    n_blocks_time = 1,  #size,
    )
    (obs_dipole_4,) = sim.create_observations(detectors= dets,
    n_blocks_det = 1,
    n_blocks_time = 1,  #size,
    )
    obs_dip = [obs_dipole_0,obs_dipole_4]
    dipole_type = [0,4]
    orbit = lbs.SpacecraftOrbit(obs_dipole_0.start_time)
    pos_vel = lbs.spacecraft_pos_and_vel(orbit, obs_dipole_0, delta_time_s=86400.0)
    for i_ob,ob in enumerate(obs_dip):
        lbs.add_dipole_to_observations(ob, pointings, pos_vel, dipole_type=dipole_type[i_ob])
    t_dip = time.time()
    print('time for dipole construction: ', t_dip-t_noise)

    
    obs=obs+obs_noise+obs_dip
    obs_name = map_type+noisetype+['linear','total_from_lin_T']
    descr = ['_','_','_','_noise_','_noise_','_dip_','_dip_']

    for o_i,o_n in enumerate(obs_name):
        file_name = base_path+'/obs'+descr[o_i]+o_n+'.h5'
        with h5py.File(file_name, "w") as output_file:
            lbs.io.write_one_observation(
                        output_file=output_file,
                        obs=obs[o_i],
                        tod_dtype=np.float32,
                        pointings_dtype=np.float32,
                    )

    t_save = time.time()
    print('time for saving tods: ', t_save-t_dip)
