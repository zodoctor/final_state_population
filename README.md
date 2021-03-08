# Black Hole Leftovers

## Basic Idea:
The masses and spins of two inspiraling and merging black holes
determines the mass, spin, and kick velocity of the remnant
black hole.  Therefore, knowing the mass and spin distribution
of the inspiraling bodies means knowing the distribution of
the final state mass and spin and kick velocity.  This 
can be straightforwardly calculated from the hyperparameter posterior
samples from the LVC.  We provide results for the mass, spin
and final velocity spectra and number densities of remant BHs

## Accessing the Data
It's straightforward to re-run the `Black_Hole_Leftovers.ipynb` notebook if
you want to change any aspects of the analysis or make your own plots. Just make
sure to grab the LVC populations paper data release folder called `Publication_Samples.tar.gz` from [here](https://dcc.ligo.org/LIGO-P2000434/public).
You can skip some of the expensive steps using precomuputed quantities in
the `*.npy` files in the repo. If you just want the results,
you need to load in the `*.npy` files only.  Here's what's in these:
* `final_params.npy`: this file contains the remnant parameters for the mergers in the fiducial reference population.  It's a 2d array, with one dimension of length 3 (final mass, final spin, final velocity, in that order) and the other dimension being number of samples from the reference population.
* `fid_pop_samples.npy`: samples of the reference population, with quantities like `"mass_ratio"`
* `hyp_idx.npy`: indices of the LVC population hyperparameter samples used to calculate weights.  This is needed since we downsample the LVC pop hyperparameter samples for computational speedup.
* `fid_weights.npy`: weights of the reference population samples
* `wts_arr.npy`: weights of hyperparameter samples divided by weights of fiducial population samples.  This file in conjunction with `final_params.npy` is bare minimum you need for the remnant population calculation.  `wts_arr` is a 2d array with one dimension the same as `hyp_idx` and the other the dimension of `fid_pop_samples`.  Applying the weights from `wts_arr` to `final_params` yields weighted samples for each population hyperparameter sample. 
* `max_L_idx.npy` is the index of the hyperparameter sample with highest likelihood which is the reference distribution chosen here
* `n_event_samples.npy` is the number of samples from the fiducial population used
