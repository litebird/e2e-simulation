import litebird_sim as lbs
import numpy as np
import healpy as hp
import time
from astropy.time import Time
import os
#import pickle
#import h5py
import pathlib

def e2e_sim_production(telescope, det_names_file, nside, mission_time_days, isim, base_path, mapmaking_type='destriper'):
    '''
    This function initializes a simulation, generates or reads a dictionaty of CMB/FG maps, one for each
    detector, and writes seven separated timelines (cmb,fg w/o band integration, fg w/ band integration,
    white noise, white+1/f noise, linear dipole, complete dipole) to be saved in separated hdf5 files. 
    The time employed for each step is printed.
    telescope: telescope name (string) e.g. 'LFT';
    det_names_file: file containing detector names, each one associated with its channel and noise NET.
                    Only and all detectors in this file will be used, e.g. to consider only top
                    detectors you should produce a file only with those detectors. This string is
                    also used for save file names;
    channels: list of channels (returned from read_channel_detname_noise)
              IMPORTANT: this script should be run with only 1 channel;
    detnames: list of detector names (returned from read_channel_detname_noise);
    noises: list of rescaled noise (returned from read_channel_detname_noise);
    nside: the nside for the CMB and FG maps generated;
    mission_time_days: days of observations;
    isim: simulation number;
    base_path: path where you want to save the maps and observations generated;
    mapmaking_type: binned, destriper or all
    '''

    #for parallelization; each rank handles 1 day of observation
    comm = lbs.MPI_COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #read channel, noise and detector names
    det_names_file_path = os.path.dirname(os.getcwd())+"/ancillary/"+det_names_file+".txt"
    det_file = np.genfromtxt(det_names_file_path, skip_header = 1, dtype = str)

    channels = det_file[:,1]
    noises   = det_file[:,4].astype(dtype = float)
    detnames = det_file[:,5]

    #simulation start time
    start_time = Time('2030-04-01T00:00:00')

    #number of detectors = raws of {det_names_file}.txt
    ndet = np.size(detnames)

    #simulation folder
    base_path += "e2e_ns"+str(nside)+"/sim"+str(isim).zfill(3)
    
    if(rank==0):
        t_in = time.time()

    #initializing the IMO
    imo = lbs.Imo()

    #initializing the simulation
    sim = lbs.Simulation(base_path=base_path,
                         mpi_comm=comm,
                         start_time=start_time,
                         duration_s=mission_time_days*24*3600.0
                         )

    comm.barrier()

    #loading the instrument metadata
    inst_info = sim.imo.query("/releases/v1.3/satellite/"+telescope+"/instrument_info")

    #generating the quaternions of the instrument 
    sim.generate_spin2ecl_quaternions(imo_url="/releases/v1.3/satellite/scanning_parameters/")

    #loading instrument info
    inst = lbs.InstrumentInfo(name=telescope, 
                              boresight_rotangle_rad=np.deg2rad(inst_info.metadata["boresight_rotangle_deg"]),
                              spin_boresight_angle_rad=np.deg2rad(inst_info.metadata["spin_boresight_angle_deg"]),
                              spin_rotangle_rad=np.deg2rad(inst_info.metadata["spin_rotangle_deg"])
                              )
    
    #filling dets with info and detquats with quaternions of the detectors in detlist
    dets = []
    detquats = []
    for i_det in range(ndet):
        det = lbs.DetectorInfo.from_imo(url="/releases/v1.3/satellite/"+telescope+"/"+channels[i_det]+"/"+detnames[i_det]+"/detector_info",
                                        imo=imo)
        #det.sampling_rate_hz = 1 #if commented, this parameter is taken from the IMO
        dets.append(det)
        detquats.append(det.quat)

    if(rank==0):
        t_sim = time.time()
        print('simulation time: ',t_sim - t_in)

    #initialize and fill 1/f pessimistic and 1/f realistic noise observations (each containing also white noise)
    # pessimistic: initialization
    (obs_noise_w_1_f_pessimistic,) = sim.create_observations(detectors=dets,
                                                             n_blocks_det=1,
                                                             n_blocks_time=size,
                                                             split_list_over_processes=False,
                                                             )
    # pessimistic: set knee frequency and noise specification
    obs_noise_w_1_f_pessimistic.fknee_mhz = 100
    obs_noise_w_1_f_pessimistic.net_ukrts = noises

    # pessimistic: add noise
    lbs.add_noise_to_observations([obs_noise_w_1_f_pessimistic],
                                  'one_over_f',
                                  scale=1)

    # realistic: initialization
    (obs_noise_w_1_f_realistic,) = sim.create_observations(detectors=dets,
                                                           n_blocks_det=1,
                                                           n_blocks_time=size,
                                                           split_list_over_processes=False,
                                                           )
    # realistic: set knee frequency and noise specification
    obs_noise_w_1_f_realistic.fknee_mhz = 30
    obs_noise_w_1_f_realistic.net_ukrts = noises

    # realistic: add noise
    lbs.add_noise_to_observations([obs_noise_w_1_f_realistic],
                                  'one_over_f',
                                  scale=1)
    
    if(rank==0):
        t_noise_1_f = time.time()
        print('time for filling 1/f noise timeline: ', t_noise_1_f-t_sim)

    #initialize and fill white noise observation
    (obs_noise_w,) = sim.create_observations(detectors= dets,
                                             n_blocks_det = 1,
                                             n_blocks_time = size,
                                             split_list_over_processes=False,
                                             )
    obs_noise_w.net_ukrts = noises
    lbs.add_noise_to_observations([obs_noise_w],
                                  'white',
                                  scale=1)

    if(rank==0):
        t_noise = time.time()
        print('time for filling %s noise timeline: '%('white'), t_noise-t_noise_1_f)

    #create list of noise observations
    obs_noise = [obs_noise_w,obs_noise_w_1_f_pessimistic,obs_noise_w_1_f_realistic]

    #initialize cmb and fg observations
    (obs_cmb,) = sim.create_observations(detectors=dets,
                                         n_blocks_det=1,
                                         n_blocks_time=size,
                                         split_list_over_processes=False,
                                         )

    (obs_fg,) = sim.create_observations(detectors=dets,
                                        n_blocks_det=1,
                                        n_blocks_time=size,
                                        split_list_over_processes=False,
                                        )

    if(rank==0):
        t_obs = time.time()
        print('time for obs initialization: ', t_obs-t_noise)

    #hwp specification
    hwp_radpsec = inst_info.metadata["hwp_rpm"]*2*np.pi/60

    #get pointings and store them in obs_cmb
    pointings = lbs.pointings.get_pointings(obs_cmb,
                                            spin2ecliptic_quats = sim.spin2ecliptic_quats,
                                            detector_quats = detquats,
                                            bore2spin_quat = inst.bore2spin_quat,
                                            hwp = lbs.IdealHWP(hwp_radpsec),   #applies hwp rotation angle to the polarization angle                                  
                                            store_pointings_in_obs=True,       #if True, stores colatitude and longitude in obs_cmb.pointings,
                                                                               #and the polarization angle in obs_cmb.psi
                                            )

    if(rank==0):
        t_point = time.time()
        print('time for pointings: ', t_point-t_obs)

    #read cmb and fg maps
    obs              = [obs_cmb,obs_fg]
    input_map_type   = ['lens_cmb','all_fg']
    input_map_folder = ['cmb/'+str(isim).zfill(2)+'/','all_fg/']

    for i_m in range(len(input_map_type)):
        #rank 0 reads maps and broadcasts them to the other processors
        if(rank==0):
            input_maps_path = '/global/cfs/cdirs/litebird/simulations/maps/post_ptep_inputs_20220522/beam_convolved/'

            #select frequency (IMPORTANT: this script should be run with only 1 channel)
            freq = int(channels[0][3:6]) #e.g.: channels[0] = 'L2-050' --> freq = 50
            #load maps
            try:
                if freq in [68,78,89]:
                    if(channels[0][:2]=='L1' or channels[0][:2]=='L2'):
                        maps = hp.read_map(input_maps_path+input_map_folder[i_m]+'LB_'+telescope+'_'+str(freq)+'a_'+input_map_type[i_m]+'_postPTEP20220609.fits',
                                           field=[0,1,2])
                    else:
                        maps = hp.read_map(input_maps_path+input_map_folder[i_m]+'LB_'+telescope+'_'+str(freq)+'b_'+input_map_type[i_m]+'_postPTEP20220609.fits',
                                           field=[0,1,2])
                else:
                    maps = hp.read_map(input_maps_path+input_map_folder[i_m]+'LB_'+telescope+'_'+str(freq)+ '_'+input_map_type[i_m]+'_postPTEP20220609.fits',
                                       field=[0,1,2])
            except:
                print("Error while reading map",input_map_type[i_m],"for channel",channels[0])
        else:
            maps = None

        #broadcast maps read by rank 0
        maps = comm.bcast(maps, root=0)

        comm.barrier()

        #fill the TOD
        lbs.scan_map_in_observations(obs[i_m],
                                     maps,
                                     pointings, #not needed if pointing already stored in obs
                                     input_map_in_galactic=False,
                                     )

    if(rank==0):
        t_tod = time.time()
        print('time for creating and scanning map tod: ', t_tod-t_point)

    comm.barrier()

    #save TODs only for simulation 0
    if(isim==0):
        #dipole (saved only as TOD for sim 0)
        (obs_dip,) = sim.create_observations(detectors=dets,
                                             n_blocks_det=1,
                                             n_blocks_time=size,
                                             split_list_over_processes=False,
                                             )
        orbit = lbs.SpacecraftOrbit(obs_dip.start_time)

        #spacecraft position and velocity
        pos_vel = lbs.spacecraft_pos_and_vel(orbit,
                                             obs_dip,
                                             delta_time_s=86400.0
                                             )

        #add dipole to obs_dip; dipole type is TOTAL_FROM_LIN_T, read the doc for more info
        lbs.add_dipole_to_observations(obs=obs_dip,
                                       pos_and_vel=pos_vel,
                                       pointings=pointings,
                                       dipole_type=lbs.DipoleType.TOTAL_FROM_LIN_T,
                                       )

        if(rank==0):
            t_dip = time.time()
            print('time for dipole construction: ', t_dip-t_tod)

        #create save path for observation
        obs_path = base_path+'/TOD_'+det_names_file
        if(rank==0):
            #this has to be done by rank 0 to avoid conflicts        
            if not os.path.exists(obs_path):
                os.mkdir(obs_path)

        #list of all the observations for which we want to save the TODs
        obs=obs+obs_noise+[obs_dip]

        custom_dicts = [
                { "myvalue": "cmb_day"+str(rank).zfill(4) }, #obs_cmb will also have the pointing saved
                { "myvalue": "fg_day"+str(rank).zfill(4) },
                { "myvalue": "w_noise_day"+str(rank).zfill(4) },
                { "myvalue": "1_over_f_noise_pessimistic_day"+str(rank).zfill(4) },
                { "myvalue": "1_over_f_noise_realistic_day"+str(rank).zfill(4) },
                { "myvalue": "dipole_total_day"+str(rank).zfill(4) },
            ]

        lbs.io.write_list_of_observations(obs=obs,
                                          path=obs_path,
                                          file_name_mask="obs_{myvalue}.hdf5",
                                          custom_placeholders=custom_dicts,
                                          collective_mpi_call=True,
                                          )

        if(rank==0):
            t_save = time.time()
            print('time for saving tods: ', t_save-t_dip)

    #create save path for output maps
    map_path = base_path+'/maps_'+det_names_file+'/'
    if(rank==0):
        if not os.path.exists(map_path):
            os.mkdir(map_path)

    comm.barrier()

    obs_list_mapmaking  = [[obs_cmb,obs_fg,obs_noise_w],                 #cmb, foregrounds and white noise #MBNR we do not need to destripe here, right?
                           [obs_cmb,obs_fg,obs_noise_w_1_f_pessimistic], #cmb, foregrounds and 1/f noise (containing also white noise) in the pessimistic case
                           [obs_noise_w_1_f_pessimistic],                #1/f noise (containing also white noise) in the pessimistic case
                           [obs_cmb,obs_fg,obs_noise_w_1_f_realistic],   #cmb, foregrounds and 1/f noise (containing also white noise) in the realistic case
                           [obs_noise_w_1_f_realistic]]                  #1/f noise (containing also white noise) in the realistic case
    
    pointings_mapmaking = [[pointings,pointings,pointings],              #MBNR: pointings_mapmaking needed? Probably no
                           [pointings,pointings,pointings],
                           [pointings],
                           [pointings,pointings,pointings],
                           [pointings]]
    
    filenames_mapmaking = ['cmb_fg_wn',
                           'cmb_fg_1fpess',
                           '1fpess',
                           'cmb_fg_1frea',
                           '1frea']

    #build the output maps with a binned mapmaker
    if(mapmaking_type=='binned' or mapmaking_type=='all'):
        for i in range(len(obs_list_mapmaking)):
            map_output = lbs.make_bin_map(obs_list_mapmaking[i],
                                          nside,
                                          pointings=pointings_mapmaking[i],
                                          do_covariance=False,
                                          output_map_in_galactic=True
                                          )
            #save binned maps
            if(rank==0):
                hp.write_map(map_path+'map_binned_'+filenames_mapmaking[i]+'_'+str(mission_time_days)+'d.fits',
                             map_output,
                             overwrite=True
                             )

    #build the output maps with a destriper
    if(mapmaking_type=='destriper' or mapmaking_type=='all'):
        for i in range(len(obs_list_mapmaking)):
            param_noise_madam = lbs.DestriperParameters(nside=nside,
                                                        nnz=3, #compute I, Q, and U
                                                        baseline_length_s=60,
                                                        return_hit_map=False,
                                                        return_binned_map=False,
                                                        return_destriped_map=False,
                                                        coordinate_system=lbs.coordinates.CoordinateSystem.Galactic,
                                                        #iter_max=10, #defaul is 100
                                                        output_file_prefix='map_destriper_'+filenames_mapmaking[i]+'_'+str(mission_time_days)+'d_'
                                                        )
            result = lbs.destriper.destripe_observations(observations=obs_list_mapmaking[i],
                                                         base_path=pathlib.PosixPath(map_path),
                                                         params=param_noise_madam,
                                                         pointings=pointings_mapmaking[i]
                                                         )

    if(rank==0):
        print("Done")



# OTHER STUFF

# def read_all_obs(telescope, detname_T, nside, mission_time_days, T_and_B, base_path):
#     """
#     This function reads and returns the observations for cmb,fg w/o band integration, fg w/ band integration,
#     white noise, white+1/f noise, linear dipole and complete dipole, saved in separated hdf5 files. 
#     telescope: telescope name (string) e.g. 'LFT';
#     detname_T: list of (Top) detector names (returned from read_channel_detname_noise);
#     nside: the nside for the CMB and FG maps generated;
#     mission_time_days: days of observations;
#     T_and_B: True (if you want to include both Top&Bottom detectors between ni_det and nf_det) or False (if you want to include just Top detector);
#     base_path: path where you want to save the maps and observations generated 
#     """
#     base_path += "/sim_ns"+str(nside)+'_'+telescope
#     ndet = np.size(detname_T)
#     if T_and_B:
#         if ndet == 1:
#             obs_path = base_path+'/det_'+detname_T[0][:-1]+'T+B_'+str(mission_time_days)+'d'
#         else:
#             obs_path = base_path+"/dets_"+'_'.join([d[:-1]+'T+B_' for d in detname_T])+'_'+str(mission_time_days)+'d'
#     else:
#         if ndet == 1:
#             obs_path = base_path+'/det_'+detname_T[0]+'_'+str(mission_time_days)+'d'
#         else:
#             obs_path = base_path+"/dets_"+'_'.join([d for d in detname_T])+'_'+str(mission_time_days)+'d'
#     print('obs. path:', obs_path)
#    
#     obs_cmb,obs_fg,obs_fg_int,obs_w_noise,obs_1_f_noise,obs_dip_lin,obs_dip_tot_linT=lbs.io.read_list_of_observations(file_name_list = [
#                                             obs_path+"/obs_cmb.hdf5",obs_path+"/obs_fg.hdf5",
#                                             obs_path+"/obs_fg_int.hdf5",obs_path+"/obs_w_noise.hdf5",
#                                             obs_path+"/obs_1_over_f_noise.hdf5",obs_path+"/obs_dip_linear.hdf5",
#                                             obs_path+"/obs_dip_total_from_lin_T.hdf5",])  
#
#
#     return obs_cmb,obs_fg,obs_fg_int,obs_w_noise,obs_1_f_noise,obs_dip_lin,obs_dip_tot_linT



# def build_map(obs, pointings, psi, detname_T, nside):
#     """
#     This function generates a map out of the observation including a specific timeline (cmb, noise, fg...). 
#     obs: one of the observations returned from read_all_obs, containing the timeline from which you want to generate a map;
#     pointings: it can be get from obs_cmb.pointings;
#     psi: it can be get from obs_cmb.psi;
#     detname_T: list of (Top) detector names (returned from read_channel_detname_noise);
#     nside: the nside of the map you want to generate;
#     """
#     ndet = np.size(detname_T)
#     try: 
#         obs.__getattribute__("psi")
#     except:
#         obs.psi = psi
#
#     obs.pixind = np.empty_like(obs.tod, dtype=np.int)
#     for i_det in range(ndet):
#         obs.pixind[i_det] = hp.ang2pix(nside, pointings[i_det, :, 0], pointings[i_det, :, 1]) 
#
#     m = lbs.make_bin_map([obs],nside,pointings=pointings)
#
#     return m



#code for simulating maps that are scanned (instead of loading them):
    # #loading channel info
    # ch_info = []
    # ch_names = ''
    # file_list = os.listdir(base_path)
    # for i_det in range(ndet):
    #     if channel[i_det] != channel[i_det-1] or i_det == 0:
    #         ch_info.append(lbs.FreqChannelInfo.from_imo(url="/releases/v1.3/satellite/"
    #                                        +telescope+"/"+channel[i_det]+"/channel_info",imo=imo))
    #         ch_names += '_'+channel[i_det]
    #         m_files = [fn for fn in file_list if str(channel[i_det]) in fn]
    #
    # #create CMB, foregrounds and band-integrated foregrounds maps + fill their TODs
    # M_cmb     = [True,False,False]
    # M_fg      = [False,True,True]
    # M_bandint = [False,False,True]
    # obs       = [obs_cmb,obs_fg,obs_fg_bandint]
    # map_type  = ['cmb','fg','fg_int']
    #
    # for i_m in range(len(M_cmb)):
    #     map_path = base_path+'/'+map_type[i_m]+'_channels'+ch_names+'.pickle'
    #     m_file = [fn for fn in m_files if map_type[i_m]+'_channels' in fn] #are there maps for all of my channels?
    #
    #     if(rank==0):
    #         print('existing maps:', m_file, 'desired map', map_path)
    #
    #     if m_file == []:   #if there are not existing maps for all of my channels (or more), I generate them
    #         #this sets the parameters for the generation of the map
    #         Mbsparams = lbs.MbsParameters(
    #             make_cmb  = M_cmb[i_m],
    #             make_fg   = M_fg[i_m],
    #             seed_cmb  = 1,
    #             fg_models = ["pysm_synch_0", "pysm_freefree_1","pysm_dust_0"], #set the FG models you want
    #             gaussian_smooth = True,        #if True, smooths the input map by the beam of the channel
    #             bandpass_int = M_bandint[i_m], #if True, integrates over the top-hat bandpass of the channel
    #             nside = nside,
    #             units = "K_CMB",
    #             maps_in_ecliptic = True,   #maps saved in ecliptic, because of dipole 
    #             )
    #
    #         mbs = lbs.Mbs(simulation = sim,
    #             parameters = Mbsparams,
    #             channel_list = ch_info,
    #             #detector_list = dets   #use detector_list instead of channel_list if your sim has detectors 
    #                                     #from different channels. It would produce a map for each detector in dets
    #             )
    #
    #         #generate the map as a dictionary
    #         maps = mbs.run_all()[0]
    #
    #         with open(map_path, 'wb') as f:
    #             pickle.dump(maps, f)
    #
    #         if(rank==0):
    #             t_map = time.time()
    #             if i_m == 0:
    #                 print('time for creating map: ', t_map-t_point)
    #             else:
    #                 print('time for creating map: ', t_map-t_tod)
    #
    #     else:   
    #         with open(base_path+'/'+m_file[0],'rb') as f:  #read the existing map
    #             read_map = pickle.load(f)
    #         if os.path.isfile(map_path):  #if the existing map includes exactly all my channels and nothing more, I just read it
    #             maps = read_map
    #         else:  #if the esisting map has more than the needed channels, I create a new dictionary with only my channels
    #             maps = {k: read_map[k] for k in read_map.keys() if k in channel}
    #
    #         if(rank==0):
    #             print('channels of our map dict.:', maps.keys())
    #         if(rank==0):
    #             t_map = time.time()
    #             if i_m == 0:
    #                 print('time for reading map: ', t_map-t_point)
    #             else:
    #                 print('time for reading map: ', t_map-t_tod)
