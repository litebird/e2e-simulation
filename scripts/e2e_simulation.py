import litebird_sim as lbs
import numpy as np
import healpy as hp
import matplotlib.pylab as plt
from astropy.time import Time
import time
import pathlib
import os
import sys

def plot_map(m,title,save_filename):
    '''
    This function saves the mollview of the map m as a png figure and returns a tuple used to
    insert the image in the report.

    m: map to be plotted;
    title: string, shown in the title of the figure;
    base_path: string, same parameter of e2e_sim_production;
    save_filename: string, name of the output file, e.g. 'my_figure.png'
    '''

    fig = plt.figure()
    hp.mollview(m,title=title,fig=fig)
    return (fig, save_filename)

def save_append_maps(map_path,map_name,map_output,cov_output,figures):
    '''
    This function saves a set of T,Q,U maps (.fits), its covariance (.npy) and
    updates the figures list with its mollweide projection.

    map_path: path where to save maps, i.e. base_path+'maps/';
    map_name: name of the map, i.e. case under study;
    map_output: map to be saved;
    cov_output: cov to be saved;
    figures: list with tuples of (fig, save_filename) that has to be updated

    returns: updated figures list
    '''
    
    #save maps
    hp.write_map(map_path+map_name+'.fits',
                 map_output,
                 overwrite=True)

    #save cov
    np.save(map_path+map_name+'_cov.npy',
            cov_output)

    #update figures list
    fields = ['T','Q','U']
    for i in range(3):
        figures.append(plot_map(map_output[i],
                                title=map_name+' - '+fields[i],
                                save_filename=map_name+'_'+fields[i]+'.png'))
    return figures

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
                         mpi_comm=comm)

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
                             dtype=str)

    channels = det_file[:,1]
    noises   = det_file[:,4].astype(dtype=float)
    detnames = det_file[:,5]

    #get frequency (IMPORTANT: this script should be run with only 1 channel)
    freq = int(channels[0][3:6]) #e.g.: channels[0] = 'L2-050' --> freq = 50

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
                              spin_rotangle_rad=np.deg2rad(inst_info.metadata["spin_rotangle_deg"]))
    
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

    #create Observation object
    (obs_multitod,) = sim.create_observations(detectors=dets,
                                              n_blocks_det=1,
                                              n_blocks_time=size,
                                              split_list_over_processes=False)
    #create arrays to store all the TODs
    #obs_multitod.tod not used #MBNR
    obs_multitod.tod_wn_1f_100mHz = np.zeros_like(obs_multitod.tod)
    obs_multitod.tod_wn_1f_30mHz  = np.zeros_like(obs_multitod.tod)
    obs_multitod.tod_wn           = np.zeros_like(obs_multitod.tod)
    obs_multitod.tod_cmb          = np.zeros_like(obs_multitod.tod)
    obs_multitod.tod_fg           = np.zeros_like(obs_multitod.tod)
    if(isim==0):
        obs_multitod.tod_dip      = np.zeros_like(obs_multitod.tod)

    #pessimistic 1/f: set knee frequency and noise specification
    obs_multitod.fknee_mhz = 100
    obs_multitod.fmin_hz   = 1e-5
    obs_multitod.net_ukrts = noises

    #pessimistic 1/f: add noise
    lbs.add_noise_to_observations([obs_multitod],
                                  'one_over_f',
                                  scale=1,
                                  component="tod_wn_1f_100mHz")

    #realistic 1/f: set knee frequency
    obs_multitod.fknee_mhz = 30

    #realistic 1/f: add noise
    lbs.add_noise_to_observations([obs_multitod],
                                  'one_over_f',
                                  scale=1,
                                  component="tod_wn_1f_30mHz")
    
    if(rank==0):
        t_noise_1_f = time.time()
        print('time for filling 1/f noise timeline: ', t_noise_1_f-t_sim)

    #white noise: add noise
    lbs.add_noise_to_observations([obs_multitod],
                                  'white',
                                  scale=1,
                                  component="tod_wn")

    if(rank==0):
        t_noise = time.time()
        print('time for filling %s noise timeline: '%('white'), t_noise-t_noise_1_f)

    #hwp specification
    hwp_radpsec = inst_info.metadata["hwp_rpm"]*2*np.pi/60

    #get pointings and store them in obs_multitod
    pointings = lbs.pointings.get_pointings(obs_multitod,
                                            spin2ecliptic_quats=sim.spin2ecliptic_quats,
                                            detector_quats=detquats,
                                            bore2spin_quat=inst.bore2spin_quat,
                                            hwp=lbs.IdealHWP(hwp_radpsec),   #applies hwp rotation angle to the polarization angle                                  
                                            store_pointings_in_obs=True)     #if True, stores colatitude and longitude in obs_multitod.pointings,
                                                                             #and the polarization angle in obs_multitod.psi

    if(rank==0):
        t_point = time.time()
        print('time for pointings: ', t_point-t_noise)

    #read and scan cmb and fg maps
    input_map_type   = ['lens_cmb',                    'all_fg']
    input_map_folder = ['cmb/'+str(isim).zfill(2)+'/', 'all_fg/']
    comp             = ['tod_cmb',                     'tod_fg']

    for i_m in range(len(input_map_type)):
        #rank 0 reads maps and broadcasts them to the other processors
        if(rank==0):
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
        
        #convert from uK to K #MBNR hard coded...
        maps *= 1e-6

        comm.barrier()

        #fill the TOD
        lbs.scan_map_in_observations(obs_multitod,
                                     maps,
                                     #pointings, #not needed if pointing already stored in obs
                                     input_map_in_galactic=True,
                                     component=comp[i_m])

    if(rank==0):
        t_tod = time.time()
        print('time for reading and scanning map for TODs: ', t_tod-t_point)

    comm.barrier()

    #produce dipole TODs and save all TODs only for simulation 0
    if(isim==0):
        #dipole (saved only as TOD for sim 0)
        orbit = lbs.SpacecraftOrbit(obs_multitod.start_time)

        #spacecraft position and velocity
        pos_vel = lbs.spacecraft_pos_and_vel(orbit,
                                             obs_multitod,
                                             delta_time_s=86400.0)

        #add dipole to obs_multitod; dipole type is TOTAL_FROM_LIN_T, read the doc for more info
        lbs.add_dipole_to_observations(obs=obs_multitod,
                                       pos_and_vel=pos_vel,
                                       pointings=pointings,
                                       dipole_type=lbs.DipoleType.TOTAL_FROM_LIN_T,
                                       component='tod_dip')

        if(rank==0):
            t_dip = time.time()
            print('time for dipole construction: ', t_dip-t_tod)

        #create save path for observation
        obs_path = base_path+'tods/'
        if(rank==0):
            #this has to be done by rank 0 to avoid conflicts        
            if not os.path.exists(obs_path):
                os.mkdir(obs_path)

        #create list for components to be saved
        if(isim==0):
            field_list = ['tod_cmb','tod_fg','tod_dip','tod_wn','tod_wn_1f_100mHz','tod_wn_1f_30mHz']
        else:
            field_list = ['tod_cmb','tod_fg',          'tod_wn','tod_wn_1f_100mHz','tod_wn_1f_30mHz']
        
        #save tods
        tod_out_filename_dict = [{ "myvalue": "LB_"+telescope+"_"+str(freq)+"_obs_rank"+str(rank).zfill(4) }]
        
        lbs.io.write_list_of_observations(obs=obs_multitod,
                                          path=obs_path,
                                          file_name_mask="{myvalue}.hdf5",
                                          custom_placeholders=tod_out_filename_dict,
                                          collective_mpi_call=True,
                                          tod_fields=field_list)

        if(rank==0):
            t_save_tod = time.time()
            print('time for saving tods: ', t_save_tod-t_dip)

    #create save path for output maps
    map_path = base_path+'maps/'
    if(rank==0):
        if not os.path.exists(map_path):
            os.mkdir(map_path)

    comm.barrier()    

    if(rank==0):
        figures = []

    #create lists of combined components for mapmakers...
    obs_list_mapmaking = [['tod_cmb', 'tod_fg', 'tod_wn'],
                          ['tod_cmb', 'tod_fg', 'tod_wn_1f_100mHz'],
                          ['tod_cmb', 'tod_fg', 'tod_wn_1f_30mHz'],
                          ['tod_wn_1f_100mHz'],
                          ['tod_wn_1f_30mHz']]
    #...and of their names
    obs_name_mapmaking = ['cmb_fg_wn',
                          'cmb_fg_wn_1f_100mHz',
                          'cmb_fg_wn_1f_30mHz',
                          'wn_1f_100mHz',
                          'wn_1f_30mHz']

    save_files = True #flag for saving tods and pointings only once with save_simulation_for_madam

    for obs_list,obs_name in zip(obs_list_mapmaking,obs_name_mapmaking):
        
        #binned mapmaker
        if(mapmaking_type=='binned' or mapmaking_type=='all'):
            #build the output maps
            map_output, cov_output = lbs.make_bin_map(obs_multitod,
                                                      nside,
                                                      do_covariance=True,
                                                      output_map_in_galactic=True,
                                                      components=obs_list)
            #save maps and create figures for report
            if(rank==0):
                map_name = 'LB_'+telescope+'_'+str(freq)+'_binned_'+obs_name+'_'+mission_time_days+'d'+'_'+str(isim).zfill(4)
                figures = save_append_maps(map_path,
                                           map_name,
                                           map_output,
                                           cov_output,
                                           figures)
            comm.barrier()
        
        #destriper
        if(mapmaking_type=='destriper' or mapmaking_type=='all'):
            #no destriping without 1/f noise
            if(obs_name=='cmb_fg_wn'):
                continue
            #set destriper parameters
            params_madam = lbs.DestriperParameters(nside=nside,
                                                   coordinate_system=lbs.coordinates.CoordinateSystem.Galactic,
                                                   nnz=3, #compute I, Q, and U
                                                   baseline_length_s=60,
                                                   iter_max=100, #default is 100
                                                   return_hit_map=True,
                                                   return_binned_map=True,
                                                   return_destriped_map=True,
                                                   return_npp=False,
                                                   return_invnpp=False,
                                                   return_rcond=False)
            #save results to be read by madam
            lbs.madam.save_simulation_for_madam(sim=sim,
                                                params=params_madam,
                                                detectors=dets,
                                                use_gzip=False,
                                                #output_path=base_path, #default is sim.base_path / "madam_subfolder_name"
                                                absolute_paths=True,
                                                madam_subfolder_name='madam_'+obs_name,
                                                components=['tod_cmb','tod_fg','tod_wn_1f_100mHz','tod_wn_1f_30mHz'],
                                                components_to_bin=obs_list,
                                                save_pointings=save_files,
                                                save_tods=save_files)

            save_files = False #for saving tods and pointings only once with save_simulation_for_madam

            comm.barrier()

    if(rank==0):
        t_save_maps = time.time()
        if(isim==0):
            print('time for saving tods: ', t_save_maps-t_save_tod)
        else:
            print('time for saving tods: ', t_save_maps-t_tod)

        # Create report
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

## Output maps

Produced output maps:

{% for figure in figs %}
 ![]({{ figure[1] }})
{% endfor %}

## Detector list

Detectors used in the simulation:

{% for detname in detnames %}
 `{{ detname }}`
{% endfor %}

## How to read the output

### Maps and covariances

```python
import healpy
m = healpy.read_map("path/to/file.fits", field=[0, 1, 2])
```

### Covariances in NPY format

```python
import numpy as np
cov = np.load("path/to/filename.fits")
```

### TODs and pointings (observations)

```python
import litebird_sim as lbs

obs = lbs.io.read_one_observation("path/to/file.hdf5", limit_mpi_rank=False, tod_fields=['tod_name1','tod_name2', ...])
```

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
        start_time        = start_time,
        figures           = figures,
        figs              = figures, #needed to loop over figures
        detnames          = detnames)

        sim.flush()

        t_report = time.time()
        print('time for report: ', t_report-t_save_maps)

        print("Done")
