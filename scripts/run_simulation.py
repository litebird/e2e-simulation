import litebird_sim as lbs
import numpy as np
import matplotlib.pylab as plt
import healpy as hp

comm = lbs.MPI_COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = 0
mission_time_days = 1
telescope = "MFT"
channel = "M1-140"
base_path = "./tut00"
outmapfile = "test.fits"
noisetype = "white"

nside = 128

imo = lbs.Imo()

sim = lbs.Simulation(base_path=base_path,mpi_comm=comm,start_time=start_time,duration_s=mission_time_days*24*3600.0)

inst_info = sim.imo.query("/releases/v1.0/satellite/"+telescope+"/instrument_info")
sim.generate_spin2ecl_quaternions(imo_url="/releases/v1.0/satellite/scanning_parameters/")
inst = lbs.InstrumentInfo(name=telescope, 
    boresight_rotangle_rad=np.deg2rad(inst_info.metadata["boresight_rotangle_deg"]),
    spin_boresight_angle_rad=np.deg2rad(inst_info.metadata["spin_boresight_angle_deg"]),
    spin_rotangle_rad=np.deg2rad(inst_info.metadata["spin_rotangle_deg"]),)
ch_info = lbs.FreqChannelInfo.from_imo(url="/releases/v1.0/satellite/"+telescope+"/"+channel+"/channel_info",imo=imo)

hwp_radpsec = inst_info.metadata["hwp_rpm"]*2*np.pi/60

if rank == 0:
    print(ch_info.number_of_detectors)

dets=[]
detquats=[]
for detname in ch_info.detector_names:
    det=lbs.DetectorInfo.from_imo(url="/releases/v1.0/satellite/"+telescope+"/"+channel+"/"+detname+"/detector_info",imo=imo)
    dets.append(det)
    detquats.append(det.quat)
    
obs, = sim.create_observations(detectors=dets,
    split_list_over_processes = False,
    n_blocks_det = 1,
    n_blocks_time = size,
    )

pointings = lbs.get_pointings(obs,
    spin2ecliptic_quats = sim.spin2ecliptic_quats,
    detector_quats = detquats,
    bore2spin_quat = inst.bore2spin_quat,)

if rank == 0:
    Mbsparams = lbs.MbsParameters(
        make_cmb =True,
        make_fg = False,
#        fg_models = ["pysm_synch_0", "pysm_freefree_1","pysm_dust_0"],
        gaussian_smooth = True,
        bandpass_int = False,
        nside = nside,
    )
    mbs = lbs.Mbs(simulation = sim,parameters = Mbsparams,channel_list = ch_info)
    maps = mbs.run_all()[0][channel]
else:
    maps = None

comm.barrier()

maps = comm.bcast(maps, root=0)

comm.barrier()

hwp_sys = lbs.HwpSys(sim)
hwp_sys.set_parameters(maps = maps,
    nside = nside,
    integrate_in_band = False,
    built_map_on_the_fly = False,
    correct_in_solver = False,
    )

hwp_sys.fill_tod(obs,pointings,hwp_radpsec)

lbs.add_noise([obs],noisetype)

comm.barrier()
m = lbs.make_bin_map([obs],nside).T
comm.barrier()

if comm.rank==0:
    print('Writing map')
    hp.write_map(outmapfile,m,overwrite=True)
    sim.flush()
