import os
import sys
import time
from pathlib import Path
from typing import Union

#import brahmap
import healpy as hp
import litebird_sim as lbs
import matplotlib.pylab as plt
import numpy as np
from astropy.time import Time
import gzip

FG_COMPLEXITIES = {
    "low_complexity": [
        "pysm_ame_1",
        "pysm_freefree_1",
        "pysm_dust_0",
        "pysm_synch_0",
    ],
    "medium_complexity": [
        "pysm_ame_1",
        "pysm_co_1",
        "pysm_freefree_1",
        "pysm_dust_1",
        "pysm_synch_1",
    ],
    "high_complexity": [
        "pysm_ame_1",
        "pysm_co_3",
        "pysm_freefree_1",
        "pysm_dust_10",
        "pysm_synch_5",
    ],
}

NOISE_LABELS = {
    "white": "_wn",
    "one_over_f": "_wn_1f",
}


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def get_rescaled_net(detnames: list[str], chinfo: lbs.FreqChannelInfo, duration_in_years: int | float):
    used_detectors = len(detnames)
    total_number_of_detectors =  chinfo.number_of_detectors
    rescaling_factor = np.sqrt(duration_in_years/3 * used_detectors / total_number_of_detectors)
    return chinfo.net_detector_ukrts * rescaling_factor


def e2e_sim_production(
    toml_filename,
    isim,
    channel,
    seed,
):
    """
    This function reads CMB/FG maps, scans them and produces white noise, 1/f noises and cmb dipole.
    Only for the first simulation, i.e. sim0000, it writes timelines (cmb, fg, white noise,
    white noise+1/f noise with f_knee of 30mHz, white noise+1/f noise with f_knee of 100mHz, dipole)
    as hdf5 files. It then uses the timelines to produce binned and/or to save the results to then
    produce destriped maps with madam.
    The time employed for each step is printed.

    toml_filename: string, name of the TOML file where the following parameters are specified:
        imo_version: string, version of the IMO, e.g. 'v1.3';
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

    isim: int simulation number
    channel: str channel to simulate
    seed: int random seed
    """

    # for parallelization
    comm = lbs.MPI_COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        t_in = time.time()

    if rank == 0:
        print("Doing sim: " + str(isim).zfill(4))

    # initializing the IMO
    imo = lbs.Imo(flatfile_location="/dss/dssfs02/lwp-dss-0001/pn36hu/pn36hu-dss-0000/beam_sims/IMo_vPostKDP2/json1/") #FIXME: hardcoded

    sim = lbs.Simulation(
        parameter_file=toml_filename,
        mpi_comm=comm,
        random_seed=int(seed),
        imo=imo,
    )

    # more params (Marta)
    sky_path = sim.parameters["simulation"]["sky_path"]
    beam_path = sim.parameters["simulation"]["beam_path"]
    option = sim.parameters["general"]["option"]

    # extract useful parameters
    imo_location = sim.parameters["general"]["imo_location"]
    imo_version = sim.parameters["general"]["imo_version"]
    telescope = sim.parameters["general"]["telescope"]
    detectors = sim.parameters["general"]["detectors"]

    mission_time_days = sim.parameters["general"]["mission_time_days"]

    base_path = sim.parameters["simulation"]["base_path"]

    duration_s = sim.parameters["simulation"]["duration_s"]
    start_time = sim.parameters["simulation"]["start_time"]

    nside = int(sim.parameters["simulation"]["nside"])

    lmax = int(sim.parameters["simulation"]["lmax"])
    mmax = int(sim.parameters["simulation"]["mmax"])

    use_hwp = sim.parameters["simulation"]["use_hwp"]
    want_dipole_signal = sim.parameters["simulation"]["want_dipole_signal"]

    cmb_seed = sim.parameters["simulation"]["CMB_seed"]

    noise = sim.parameters["simulation"]["noise"]
    want_2f = sim.parameters["simulation"]["want_2f"]
    want_non_linearity = sim.parameters["simulation"]["want_non_linearity"]
    want_gain_drift = sim.parameters["simulation"]["want_gain_drift"]

    tod_method = sim.parameters["simulation"]["tod_method"]

    mapmaking_type = sim.parameters["simulation"]["mapmaking_type"]

    save_invcovpp = sim.parameters["simulation"]["save_invcovpp"]

    if rank == 0:
        print("Mission duration: "+mission_time_days)
        print("Simulation random seed: "+str(seed))
        print("CMB random seed: "+str(cmb_seed))
        print("IMo version: "+imo_version)

    # create new base path folder
    if rank == 0:
        if not os.path.exists(base_path):
            os.makedirs(base_path)

    # create save path for output maps
    map_path = base_path + "maps/"
    if rank == 0:
        if not os.path.exists(map_path):
            os.mkdir(map_path)

    # set instrument
    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            imo,
            f"/releases/{imo_version}/LMHFT/instrument_info",
        )
    )

    # set scanning strategy
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy.from_imo(
            url=f"/releases/{imo_version}/Observation/Scanning_Strategy",
            imo=imo,
        )
    )

    # channel
    chinfo = lbs.FreqChannelInfo.from_imo(
        url=f"/releases/{imo_version}/{telescope}/{channel}/channel_info",
        imo=imo,
    )

    # freq = chinfo.bandcenter_ghz

    if is_number(detectors):
        detnames = chinfo.detector_names[0 : int(detectors)]
    elif detectors == "all":
        detnames = chinfo.detector_names
    elif isinstance(detectors, str) and os.path.exists(detectors):
        det_names_file_path = detectors
        det_file = np.genfromtxt(det_names_file_path, skip_header=1, dtype=str)
        detnames = det_file[:, 5]
    else:
        msg = "'detectors' is neither an integer, nor the flag 'all', nor a path to a file"
        raise ValueError(msg)

    # filling dets with info and detquats with quaternions of the detectors in detlist
    dets = []
    for dn in detnames:
        det = lbs.DetectorInfo.from_imo(
            url=f"/releases/{imo_version}/{telescope}/{channel}/{dn}/detector_info",
            imo=imo,
        )

        det.net_ukrts = get_rescaled_net(detnames, chinfo, duration_in_years=float(mission_time_days)/365.25)
        dets.append(det)

    if rank == 0:
        t_sim = time.time()
        print("Time for initialization: ", t_sim - t_in)

    comm.barrier()

    # create Observation object
    sim.create_observations(
        detectors=dets,
        n_blocks_det=1,
        n_blocks_time=size,
        split_list_over_processes=False,
    )

    comm.barrier()

    # hwp specification
    if use_hwp:
        sim.set_hwp(
            lbs.IdealHWP(
                sim.instrument.hwp_rpm * 2 * np.pi / 60,
            ),  # applies hwp rotation angle to the polarization angle
        )

    sim.prepare_pointings()

    if rank == 0:
        t_point = time.time()
        print("Time for pointings: ", t_point - t_sim)

    comm.barrier()

    ###
    if sky_path != None:
        ncoeff = lbs.SphericalHarmonics.num_of_alm_from_lmax(4096,4096)                 # lmax and mmax from Luca's input alm coefficients
        sky = lbs.SphericalHarmonics(values=np.zeros((3,ncoeff)),lmax=4096,mmax=4096)
        if sim.parameters["simulation"]["want_CMB"]:
            with gzip.open(f"{sky_path}/MDR2_alm_cmb{isim}.npy.gz","rb") as f:
                sky += np.load(f, allow_pickle=True).item()
        if sim.parameters["simulation"]["want_FG"]:
            fg_model = sim.parameters["simulation"]["FG_model"]
            with gzip.open(f"{sky_path}/MDR2_alm_fg_{fg_model}_{channel}.npy.gz","rb") as f:
                sky += np.load(f, allow_pickle=True).item()        
    else: 
        Mbsparams = lbs.MbsParameters(
            make_cmb=sim.parameters["simulation"]["want_CMB"],
            make_fg=sim.parameters["simulation"]["want_FG"],
            seed_cmb=cmb_seed,
            fg_models=FG_COMPLEXITIES[sim.parameters["simulation"]["FG_model"]],
            gaussian_smooth=(True if tod_method == "scan" else False),
            bandpass_int=sim.parameters["simulation"]["want_BP_integration"],
            nside=nside,
            units="K_CMB",
            maps_in_ecliptic=False,
            store_alms=(True if tod_method == "convolution" else False),
            lmax_alms=lmax,
        )

        sky = sim.get_sky(
            parameters=Mbsparams,
            channels=None
            if sim.parameters["simulation"]["want_signal_per_detector"]
            else chinfo,
        )

    comm.barrier()
    if rank == 0:
        t_sky_generation = time.time()
        print("Time for sky generation: ", t_sky_generation - t_point)

    if tod_method == "convolution":
        
        ###
        if beam_path != None:
            blms = {}
            for i in range(len(detnames)):
                detname = detnames[i]
                blm = lbs.SphericalHarmonics.read_fits("{beam_path}/{option}/beam_{channel}_{detname}_nside1024.fits")
                blms[detname] = blm
        else:    
            blms = sim.get_gauss_beam_alms(
                lmax=lmax,
                mmax=mmax,
            )

        Convparams = lbs.BeamConvolutionParameters(
            lmax=lmax,
            mmax=mmax,
            single_precision=False,
            epsilon=1e-5,
        )

        sim.convolve_sky(
            sky_alms=sky,
            beam_alms=blms,
            convolution_params=Convparams,
        )
    else:
        sim.fill_tods(sky)

    comm.barrier()
    if rank == 0:
        t_tod = time.time()
        print("Time for signal: ", t_tod - t_sky_generation)


    # TODO! figure out correct order in which to apply effects!

    if want_dipole_signal:
        sim.add_dipole()

    comm.barrier()

    if noise:
        sim.add_noise(noise_type=noise)
        # TODO! if one_over_f is chosen, the MPI tasks may be assigned a short time chunk, on which the 1/f is not correctly described. In other words, you cut the correlation length artificially (if the number of time blocks is bigger than one, the i/f noise across time chunks is discontinuous.)

    comm.barrier()

    if want_2f:
        sim.add_2f()

    comm.barrier()

    if want_non_linearity:
        sim.apply_quadratic_nonlin()

    comm.barrier()

    if want_gain_drift:
        sim.apply_gaindrift(user_seed=sim.random_seed)
        # TODO! Same as 1/f noise, see above.


    comm.barrier()
    if rank == 0:
        t_common = time.time()
        print("Time for other components: ", t_common - t_tod)


    if mapmaking_type:
        field_names = ["I", "Q", "U"]
        if mapmaking_type in ["all", "brahmap"]:
            brahmap_inv_cov = brahmap.LBSim_InvNoiseCovLO_UnCorr(sim.observations)
            # TODO! Change operator to circulant matrix when it is available from BrahMap

        if mapmaking_type == "binned":
            map_output = sim.make_binned_map(
                nside=nside,
                )
            mapmaking_label = "_binned"
        if mapmaking_type == "brahmap":
            map_output = sim.make_brahmap_gls_map(
                nside=nside,
                inv_noise_cov_operator=brahmap_inv_cov,
            )
            mapmaking_label = "_brahmap"

        if mapmaking_type == "all":
            map_output = {}
            map_output["binned"] = sim.make_binned_map(
                nside=nside,
                )
            map_output["brahmap"] = sim.make_brahmap_gls_map(
                nside=nside,
                inv_noise_cov_operator=brahmap_inv_cov,
            )
            mapmaking_label = ["_binned", "_brahmap"]

        if save_invcovpp:
            field_names += ["II", "IQ", "IU", "QQ", "QU", "UU"]
            pass  # TODO! (we can use the same trick as in the binner, where we store the 9 elements as extra fields in the .fits file. BrahMap is still not compatible, though it will be soon)

        comm.barrier()

        if rank == 0:
            components_label = get_components_label(sim.parameters)
            if isinstance(mapmaking_label, list):
                for map_label in mapmaking_label:
                    map_name = (
                        "LB_"
                        + telescope
                        + "_"
                        + channel
                        + map_label
                        + components_label
                        + "_"
                        + mission_time_days
                        + "d"
                        + "_"
                        + str(isim).zfill(4)
                    )
                    coords = lbs.coord_sys_to_healpix_string(map_output[map_label.replace("_", "")].coordinate_system)
                    sim.write_healpix_map(
                        map_path + map_name + ".fits",
                        map_output[map_label.replace("_", "")].binned_map if map_label == "_binned" else map_output[map_label.replace("_", "")].GLS_maps,
                        column_names=field_names,
                        coord=coords,
                        overwrite=True,
                    )
            else:
                map_name = (
                    "LB_"
                    + telescope
                    + "_"
                    + channel
                    + mapmaking_label
                    + components_label
                    + "_"
                    + mission_time_days
                    + "d"
                    + "_"
                    + str(isim).zfill(4)
                )
                coords = lbs.coord_sys_to_healpix_string(map_output.coordinate_system)
                sim.write_healpix_map(
                    map_path + map_name + ".fits",
                    map_output.binned_map if mapmaking_label == "_binned" else map_output.GLS_maps,
                    column_names=field_names,
                    coord=coords,
                    overwrite=True,
                )

        comm.barrier()

        if rank == 0:
            t_maps = time.time()
            print("Time for maps: ", t_maps - t_common)

    comm.barrier()

    if rank == 0:
        t_all = time.time()
        print("Time for the entire computation: ", t_all - t_in)

def get_components_label(parameters):
    label = ""
    if parameters["simulation"]["want_CMB"]:
        label += "_cmb"
    if parameters["simulation"]["want_FG"]:
        label += "_fg"
    if parameters["simulation"]["want_BP_integration"]:
        label += "_bp"
    if parameters["simulation"]["want_dipole_signal"]:
        label += "_dipole"
    if parameters["simulation"]["tod_method"] == "scan":
        label += "_scan"
    else:
        label += "_beamconv"
    if parameters["simulation"]["noise"]:
        label += NOISE_LABELS.get(parameters["simulation"]["noise"])
    if parameters["simulation"]["want_2f"]:
        label += "_2f"
    if parameters["simulation"]["want_non_linearity"]:
        label += "_nonlin"
    if parameters["simulation"]["want_gain_drift"]:
        label += "_gaindrift"
    return label
