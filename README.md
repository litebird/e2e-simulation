# e2e-simulation

_scripts_ folder:
* `produce_det_names_files.ipynb`: jupyter notebook for producing the files with the detector information needed for `e2e_simulation.py`. All and only the detectors in each file will be used by `e2e_simulation.py`.

* `e2e_simulation.py`: this script produces the official end-to-end simulations for the LiteBIRD collaboration. This code is meant to be used with files produced by `produce_det_names_files.ipynb` and each file should contain only detectors in one channel (see examples in the _ancillary_ folder). If the simulation index is 0, TODs and maps are saved. If the simulation index is > 0, only maps are saved. The produced TODs are cmb, foregrounds, white noise, 1/f in the pessimistic case (f_knee = 100 mhz), 1/f in the realistic case (f_knee = 30 mhz) and dipole. The TODs are saved for each day. The produced maps contain:
    * cmb, fg, 1/f noise for the pessimistic case
    * 1/f noise for the pessimistic case
    * cmb, fg, 1/f noise for the realistic case
    * 1/f noise for the realistic case
    * cmb, fg, white noise

* `run_simulation.py`: example script for running `e2e_simulation.py` in a HPC facility (the example is for NERSC). Before using it, fill the #COMPLETE HERE fields

* `pos_vel.py`: script for calculating the spacecraft position and velocity over time

_ancillary_ folder:
files with the detector information needed for `e2e_simulation.py`. All and only the detectors in each file will be used by e2e_simulation.py. Each file contains detector belonging to one channel
