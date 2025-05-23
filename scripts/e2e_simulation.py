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


def e2e_sim_production(toml_filename,
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

    isimstart = int(isimstart)
    isimend = int(isimend)
    #loop over simulations
    first_time = True

    for isim in range(isimstart,isimend+1):
   
        if (rank==0):
            print('Doing sim: '+str(isim).zfill(4))

        if (first_time):
            #initializing the simulation
            sim = lbs.Simulation(
                parameter_file=os.path.dirname(os.getcwd())+"/ancillary/"+toml_filename+".toml",
                mpi_comm=comm,
                )

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

            #number of detectors = rows of {det_names_file}.txt
            ndet = np.size(detnames)

            #loading the instrument metadata
            inst_info = sim.imo.query('/releases/'+imo_version+'/satellite/'+telescope+'/instrument_info')

            sim.set_instrument(
                lbs.InstrumentInfo.from_imo(
                imo,
                f"/releases/{imo_version}/satellite/{telescope}/instrument_info",
                )
            )

            #loading instrument info
            sim.set_scanning_strategy(
                imo_url=f"/releases/{imo_version}/satellite/scanning_parameters/"
            )
    
            #filling dets with info and detquats with quaternions of the detectors in detlist
            dets = []
            for i_det in range(ndet):
                det = lbs.DetectorInfo.from_imo(
                    url='/releases/'+imo_version+'/satellite/'+telescope+'/'+channels[i_det]+'/'+detnames[i_det]+'/detector_info',
                    imo=imo,
                    )
                dets.append(det)

            if(rank==0):
                t_sim = time.time()
                print('Time for initialization: ',t_sim-t_in)

            comm.barrier()

            #create Observation object
            sim.create_observations(
                detectors=dets,
                n_blocks_det=1,
                n_blocks_time=size,
                split_list_over_processes=False,
            )

            comm.barrier()

            #hwp specification
            sim.set_hwp(
                lbs.IdealHWP(
                sim.instrument.hwp_rpm * 2 * np.pi / 60,
                ),  # applies hwp rotation angle to the polarization angle
            )

            sim.prepare_pointings()

            if(rank==0):
                t_point = time.time()
                print('Time for pointings: ', t_point-t_sim)

        #end of first_time
        if(rank==0):
            t_common = time.time()

        sim.nullify_tod()

        comm.barrier()

        if(rank==0):
            Mbsparams = lbs.MbsParameters(
                make_cmb=sim.parameters['simulation']['want_CMB'],
                make_fg=sim.parameters['simulation']['want_FG'],
                seed_cmb=sim.parameters['simulation']['CMB_seed'],
                fg_models=FG_COMPLEXITIES[sim.parameters['simulation']['FG_model']],
                gaussian_smooth=np.logical_and(
                    sim.parameters['simulation']['tod_method'] == 'scan',
                    sim.parameters['simulation']['want_beam_convolve']
                ),
                bandpass_int=sim.parameters['simulation']['want_BP_integration'],
                nside=nside,
                units="K_CMB",
                maps_in_ecliptic=False,
            )

            mbs = lbs.Mbs(
                simulation=sim,
                parameters=Mbsparams,
                detector_list = dets
            )
            maps = mbs.run_all()[0]

        comm.barrier()

        #broadcast maps read by rank 0
        maps = comm.bcast(maps, root=0)
            
        comm.barrier()

        if (
            sim.parameters['simulation']['tod_method'] == 'convolution' and
            sim.parameters['simulation']['want_beam_convolve']
        ):
            blms = lbs.generate_gauss_beam_alms(
                observation=sim.observations,
                lmax=Mbsparams.lmax_alms,
                mmax=Mbsparams.lmax_alms,
            )

            Convparams = lbs.BeamConvolutionParameters(
                lmax=Mbsparams.lmax_alms,
                mmax=Mbsparams.lmax_alms,
                single_precision=False,
                epsilon=1e-5,
            )

            alms = {}
            for k, mapp in maps.items():
                alm = hp.map2alm(mapp, lmax=Mbsparams.lmax_alms, iter=0)
                alms[k] = lbs.SphericalHarmonics(
                    values=alm,
                    lmax=Mbsparams.lmax_alms,
                    mmax=Mbsparams.lmax_alms,
                )

            sim.convolve_sky(
                sky_alms=alms,
                beam_alms=blms,
                BeamConvolutionParameters=Convparams,
            )
        else:
            sim.fill_tods(maps)

        comm.barrier()

        #which map?

        map_output = lbs.make_bin_map([obs],
                                      nside,
                                      do_covariance=False,
                                      output_map_in_galactic=True,
                                      )
            
        if(rank==0):
            print('Producing map: 100mHz')
            map_name = 'LB_'+telescope+'_'+channels[0]+'_binned_cmb_fg_wn_1f_100mHz_'+mission_time_days+'d'+'_'+str(isim).zfill(4)
            hp.write_map(map_path+map_name+'.fits',map_output,overwrite=True)

        if(rank==0):
            t_map100 = time.time()
            print('Time for 100mHz map: ', t_map100-t_common)

        comm.barrier()

        first_time = False
