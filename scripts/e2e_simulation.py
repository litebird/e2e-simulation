import litebird_sim as lbs
import numpy as np
import healpy as hp
import matplotlib.pylab as plt
from jinja2 import Environment #for multiple figures in the report
from astropy.time import Time
import time
import pathlib
import os
import sys

def save_map(m,title,base_path,save_filename):
    '''
    This function saves the mollview of the map m as a png figure and returns a tuple used to
    insert the image in the report.

    m: map to be plotted;
    title: string, shown in the title of the figure;
    base_path: string, same parameter of e2e_sim_production;
    save_filename: string, name of the output file, e.g. 'my_figure.png'
    '''
    fig = plt.figure()
    hp.mollview(m,title=title)
    plt.savefig(base_path+save_filename)
    return (fig,save_filename)

def e2e_sim_production(toml_filename):
    '''
    This function initializes a simulation, generates or reads a dictionaty of CMB/FG maps, one for each
    detector, and writes seven separated timelines (cmb,fg w/o band integration, fg w/ band integration,
    white noise, white+1/f noise, linear dipole, complete dipole) to be saved in separated hdf5 files. 
    The time employed for each step is printed.

    toml_filename: string, name of the TOML file where the following parameters are specified:
        imo_version: string, version of the IMO, e.g. 'v1.3';
        input_maps_path: string, locaton of the input maps to be scanned;
        telescope: string, telescope name (string) e.g. 'LFT';
        det_names_file: string, file containing detector names, each one associated with its channel and noise NET.
                        Only and all detectors in this file will be used, e.g. to consider only top
                        detectors you should produce a file only with those detectors. This string is
                        also used for save file names.
                        IMPORTANT: this script should be run with only 1 channel;
        nside: int, the resolution of the maps;
        isim: int, simulation number;
        mission_time_days: string, days of observation AND number of processors used, each handling 1 observation day;
        mapmaking_type: string, type of mapmaking, 'binned', 'destriper' or 'all';
        name: string, name of the simulation;
        base_path: string, path where you want to save the maps and observations generated;
        start_time: string, start time of the simulation, e.g. '2030-04-01T00:00:00';
        duration_s: string, days of observation, e.g. '365 days' (same as mission_time_days but recognized by lbs.Simulation)
    '''

    #for parallelization; each rank handles 1 day of observation
    comm = lbs.MPI_COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if(rank==0):
        t_in = time.time()

    #initializing the IMO
    imo = lbs.Imo()

    #initializing the simulation
    sim = lbs.Simulation(parameter_file=os.path.dirname(os.getcwd())+"/ancillary/"+toml_filename+".toml",
                         mpi_comm=comm
                         )

    #extract useful parameters
    imo_version       =     sim.parameters["general"]["imo_version"]
    input_maps_path   =     sim.parameters["general"]["input_maps_path"]
    telescope         =     sim.parameters["general"]["telescope"]
    det_names_file    =     sim.parameters["general"]["det_names_file"]
    nside             = int(sim.parameters["general"]["nside"])
    isim              = int(sim.parameters["general"]["isim"])
    mission_time_days =     sim.parameters["general"]["mission_time_days"]
    mapmaking_type    =     sim.parameters["general"]["mapmaking_type"]

    base_path         =     sim.parameters["simulation"]["base_path"]
    duration_s        =     sim.parameters["simulation"]["duration_s"]
    start_time        =     sim.parameters["simulation"]["start_time"]

    #check mapmaking type
    if(mapmaking_type not in ['binned','destriper','all']):
        raise ValueError("Wrong mapmaking type")

    #read channel, noise and detector names
    det_names_file_path = os.path.dirname(os.getcwd())+"/ancillary/"+det_names_file+".txt"
    det_file = np.genfromtxt(det_names_file_path,
                             skip_header=1,
                             dtype=str
                             )

    channels = det_file[:,1]
    noises   = det_file[:,4].astype(dtype=float)
    detnames = det_file[:,5]

    #number of detectors = raws of {det_names_file}.txt
    ndet = np.size(detnames)

    comm.barrier()

    #loading the instrument metadata
    inst_info = sim.imo.query("/releases/"+imo_version+"/satellite/"+telescope+"/instrument_info")

    #generating the quaternions of the instrument 
    sim.generate_spin2ecl_quaternions(imo_url="/releases/"+imo_version+"/satellite/scanning_parameters/")

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
        det = lbs.DetectorInfo.from_imo(url="/releases/"+imo_version+"/satellite/"+telescope+"/"+channels[i_det]+"/"+detnames[i_det]+"/detector_info",
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
                                     input_map_in_galactic=True
                                     )

    if(rank==0):
        t_tod = time.time()
        print('time for reading and scanning map for TODs: ', t_tod-t_point)

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
        obs_path = base_path+'tods/'
        if(rank==0):
            #this has to be done by rank 0 to avoid conflicts        
            if not os.path.exists(obs_path):
                os.mkdir(obs_path)

        #list of all the observations for which we want to save the TODs
        obs=obs+obs_noise+[obs_dip]

        custom_dicts = [
                { "myvalue": telescope+"_"+channels[0]+"_obs_cmb_day"+str(rank).zfill(4) }, #obs_cmb will also have the pointing saved
                { "myvalue": telescope+"_"+channels[0]+"_obs_fg_day"+str(rank).zfill(4) },
                { "myvalue": telescope+"_"+channels[0]+"_obs_w_noise_day"+str(rank).zfill(4) },
                { "myvalue": telescope+"_"+channels[0]+"_obs_1_over_f_noise_pessimistic_day"+str(rank).zfill(4) },
                { "myvalue": telescope+"_"+channels[0]+"_obs_1_over_f_noise_realistic_day"+str(rank).zfill(4) },
                { "myvalue": telescope+"_"+channels[0]+"_obs_dipole_total_day"+str(rank).zfill(4) },
            ]

        lbs.io.write_list_of_observations(obs=obs,
                                          path=obs_path,
                                          file_name_mask="{myvalue}.hdf5",
                                          custom_placeholders=custom_dicts,
                                          collective_mpi_call=True,
                                          )

        if(rank==0):
            t_save_tod = time.time()
            print('time for saving tods: ', t_save_tod-t_dip)

    #create save path for output maps
    map_path = base_path+'maps/'
    if(rank==0):
        if not os.path.exists(map_path):
            os.mkdir(map_path)

    comm.barrier()    

    #build the output maps with a binned mapmaker
    if(mapmaking_type=='binned' or mapmaking_type=='all'):
        
        ##### (1) cmb, foregrounds and white noise #####
        #add fg and wn to cmb TODs (element wise)
        obs_cmb.tod += (obs_fg.tod+obs_noise_w.tod)
        #produce map and cov
        map_output, cov_output = lbs.make_bin_map(obs_cmb,
                                                  nside,
                                                  pointings=pointings,
                                                  do_covariance=True,
                                                  output_map_in_galactic=True
                                                  )
        #save binned maps
        if(rank==0):
            hp.write_map(map_path+'map_'+telescope+'_'+channels[0]+'_sim'+str(isim).zfill(4)+'_binned_cmb_fg_wn_'+mission_time_days+'d.fits',
                         map_output,
                         overwrite=True
                         )
            np.save(map_path+'cov_'+telescope+'_'+channels[0]+'_sim'+str(isim).zfill(4)+'_binned_cmb_fg_wn_'+mission_time_days+'d.npy',
                    cov_output
                    )
        #subtract fg and wn to cmb TODs (element wise)
        obs_cmb.tod -= (obs_fg.tod+obs_noise_w.tod)

        ##### (2) cmb, foregrounds and 1/f noise (containing also white noise) in the pessimistic case #####
        obs_cmb.tod += (obs_fg.tod+obs_noise_w_1_f_pessimistic.tod)
        map_output, cov_output = lbs.make_bin_map(obs_cmb,
                                                  nside,
                                                  pointings=pointings,
                                                  do_covariance=True,
                                                  output_map_in_galactic=True
                                                  )
        if(rank==0):
            hp.write_map(map_path+'map_'+telescope+'_'+channels[0]+'_sim'+str(isim).zfill(4)+'_binned_cmb_fg_1fpess_'+mission_time_days+'d.fits',
                         map_output,
                         overwrite=True
                         )
            np.save(map_path+'cov_'+telescope+'_'+channels[0]+'_sim'+str(isim).zfill(4)+'_binned_cmb_fg_1fpess_'+mission_time_days+'d.npy',
                    cov_output
                    )
        obs_cmb.tod -= (obs_fg.tod+obs_noise_w_1_f_pessimistic.tod)

        ##### (3) cmb, foregrounds and 1/f noise (containing also white noise) in the realistic case #####
        obs_cmb.tod += (obs_fg.tod+obs_noise_w_1_f_realistic.tod)
        map_output, cov_output = lbs.make_bin_map(obs_cmb,
                                                  nside,
                                                  pointings=pointings,
                                                  do_covariance=True,
                                                  output_map_in_galactic=True
                                                  )
        if(rank==0):
            hp.write_map(map_path+'map_'+telescope+'_'+channels[0]+'_sim'+str(isim).zfill(4)+'_binned_cmb_fg_1frea_'+mission_time_days+'d.fits',
                         map_output,
                         overwrite=True
                         )
            np.save(map_path+'cov_'+telescope+'_'+channels[0]+'_sim'+str(isim).zfill(4)+'_binned_cmb_fg_1frea_'+mission_time_days+'d.npy',
                    cov_output
                    )
        obs_cmb.tod -= (obs_fg.tod+obs_noise_w_1_f_realistic.tod)

        ##### (4) 1/f noise (containing also white noise) in the pessimistic case #####
        map_output, cov_output = lbs.make_bin_map(obs_noise_w_1_f_pessimistic,
                                                  nside,
                                                  pointings=pointings,
                                                  do_covariance=True,
                                                  output_map_in_galactic=True
                                                  )
        if(rank==0):
            hp.write_map(map_path+'map_'+telescope+'_'+channels[0]+'_sim'+str(isim).zfill(4)+'_binned_1fpess_'+mission_time_days+'d.fits',
                         map_output,
                         overwrite=True
                         )
            np.save(map_path+'cov_'+telescope+'_'+channels[0]+'_sim'+str(isim).zfill(4)+'_binned_1fpess_'+mission_time_days+'d.npy',
                    cov_output
                    )

        ##### (5) 1/f noise (containing also white noise) in the realistic case #####
        map_output, cov_output = lbs.make_bin_map(obs_noise_w_1_f_realistic,
                                                  nside,
                                                  pointings=pointings,
                                                  do_covariance=True,
                                                  output_map_in_galactic=True
                                                  )
        if(rank==0):
            hp.write_map(map_path+'map_'+telescope+'_'+channels[0]+'_sim'+str(isim).zfill(4)+'_binned_1frea_'+mission_time_days+'d.fits',
                         map_output,
                         overwrite=True
                         )
            np.save(map_path+'cov_'+telescope+'_'+channels[0]+'_sim'+str(isim).zfill(4)+'_binned_1frea_'+mission_time_days+'d.npy',
                    cov_output
                    )

        #save outputs for madam mapmaker #MBNR to test
        #param_noise_madam = lbs.DestriperParameters(nside=nside,
        #                                            nnz=3, #compute I, Q, and U
        #                                            baseline_length_s=60,
        #                                            return_hit_map=False,
        #                                            return_binned_map=True,
        #                                            return_destriped_map=True,
        #                                            coordinate_system=lbs.coordinates.CoordinateSystem.Galactic,
        #                                            #iter_max=100, #default is 100
        #                                            output_file_prefix='map_destriper_'+filenames_mapmaking[i]+'_'+mission_time_days+'d_'
        #                                            )
        #lbs.madam.save_simulation_for_madam(sim=sim,
        #                                    detectors=dets,
        #                                    params=param_noise_madam,
        #                                    use_gzip=False,
        #                                    output_path= base_path+'madam/',
        #                                    absolute_paths=True)

    #build the output maps with a destriper #MBNR: to complete
    #if(mapmaking_type=='destriper' or mapmaking_type=='all'):
    #    if(rank==0):
    #        for i in range(len(obs_list_mapmaking)):
    #            param_noise_destriper = lbs.DestriperParameters(nside=nside,
    #                                                            nnz=3, #compute I, Q, and U
    #                                                            baseline_length_s=60,
    #                                                            return_hit_map=False,
    #                                                            return_binned_map=True,
    #                                                            return_destriped_map=True,
    #                                                            coordinate_system=lbs.coordinates.CoordinateSystem.Galactic,
    #                                                            #iter_max=100, #default is 100
    #                                                            output_file_prefix='map_destriper_'+filenames_mapmaking[i]+'_'+mission_time_days+'d_'
    #                                                            )
    #            result = lbs.destriper.destripe_observations(observations=obs_list_mapmaking[i],
    #                                                         base_path=pathlib.PosixPath(map_path),
    #                                                         params=param_noise_destriper,
    #                                                         pointings=pointings#pointings_mapmaking[i]
    #                                                         )

    comm.barrier()

    if(rank==0):
        t_save_maps = time.time()
        if(isim==0):
            print('time for saving tods: ', t_save_maps-t_save_tod)
        else:
            print('time for saving tods: ', t_save_maps-t_tod)

    # Create report
    if(rank==0):
        #Used parameters
        sim.append_to_report("""

## Run parameters

[General]

- imo_version = {{imo_version}}
- input_maps_path = `{{input_maps_path}}`
- telescope = {{telescope}}
- det_names_file = `{{det_names_file}}`
- nside = {{nside}}
- isim = {{isim}}
- mission_time_days = {{mission_time_days}}
- mapmaking_type = {{mapmaking_type}}

[Simulation]

- base_path = `{{base_path}}`
- start_time = {{start_time}}
- duration_s = {{duration_s}}
""",
        imo_version       = imo_version,
        input_maps_path   = input_maps_path,
        telescope         = telescope,
        det_names_file    = det_names_file,
        nside             = nside,
        isim              = isim,
        mission_time_days = mission_time_days,
        mapmaking_type    = mapmaking_type,
        base_path         = base_path,
        duration_s        = duration_s,
        start_time        = start_time
        )

        #Output maps
        figures = []

        if(mapmaking_type=='binned' or mapmaking_type=='all'):
            #maps produced by binned mapmaker
            figures.append(save_map(map_output[0],
                                    title='binned T map from binned mapmaker',
                                    base_path=base_path,
                                    save_filename='binned_T_map.png'
                                    ))
            figures.append(save_map(map_output[1],
                                    title='binned Q map from binned mapmaker',
                                    base_path=base_path,
                                    save_filename='binned_Q_map.png'
                                    ))
            figures.append(save_map(map_output[2],
                                    title='binned U map from binned mapmaker',
                                    base_path=base_path,
                                    save_filename='binned_U_map.png'
                                    ))

        if(mapmaking_type=='destriper' or mapmaking_type=='all'):
            #binned maps produced by destriper mapmaker
            figures.append(save_map(result.binned_map[0],
                                    title='binned T map from destriper mapmaker',
                                    base_path=base_path,
                                    save_filename='destriper_binned_T_map.png'
                                    ))
            figures.append(save_map(result.binned_map[1],
                                    title='binned Q map from destriper mapmaker',
                                    base_path=base_path,
                                    save_filename='destriper_binned_Q_map.png'
                                    ))
            figures.append(save_map(result.binned_map[2],
                                    title='binned U map from destriper mapmaker',
                                    base_path=base_path,
                                    save_filename='destriper_binned_U_map.png'
                                    ))
            #destriped maps produced by destriper mapmaker
            figures.append(save_map(result.destriped_map[0],
                                    title='destriped T map from destriper mapmaker',
                                    base_path=base_path,
                                    save_filename='destriper_destriped_T_map.png'
                                    ))
            figures.append(save_map(result.destriped_map[1],
                                    title='destriped Q map from destriper mapmaker',
                                    base_path=base_path,
                                    save_filename='destriper_destriped_Q_map.png'
                                    ))
            figures.append(save_map(result.destriped_map[2],
                                    title='destriped U map from destriper mapmaker',
                                    base_path=base_path,
                                    save_filename='destriper_destriped_U_map.png'
                                    ))

        #loop over list of tuples
        TEMPLATE = """
## Output maps

Produced output maps:

{% for figure in figures %}
 ![]({{ figure[1] }})
{% endfor %}
"""
        template = Environment().from_string(TEMPLATE)
        sim.append_to_report(template.render(figures=figures))

        #Detector list
        sim.append_to_report("""
## Detector list

Detectors used in the simulation:

{% for detname in detnames %}
 `{{ detname }}`
{% endfor %}

""",
        detnames = detnames,
        )

        sim.flush()

        t_report = time.time()
        print('time for report: ', t_report-t_save_maps)

        print("Done")
