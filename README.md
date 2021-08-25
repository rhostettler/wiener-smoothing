# Particle smoothing for Wiener state-space models
This is a Matlab implementation of the two filter particle smoothers for 
Wiener state space models from the following two publications:

1. R. Hostettler, “A two filter particle smoother for Wiener state-space 
   systems,” in *IEEE Conference on Control Applications (CCA)*, Sydney, 
   Australia, September 2015
  
   [[Link](http://dx.doi.org/10.1109/CCA.2015.7320664)] [[PDF](http://hostettler.co/assets/publications/msc2015.pdf)]

2. R. Hostettler and T. B. Schön, “Auxiliary-particle-filter-based two-
   filter smoothing for Wiener state-space models,” in *21th International 
   Conference on Information Fusion (FUSION)*, Cambridge, UK, July 2018
  
   [[Link](https://dx.doi.org/10.23919/ICIF.2018.8455323]) [[PDF](http://hostettler.co/assets/publications/2018-fusion-ps.pdf)]

The examples are implemented in the following files:

* `example_lgss.m`: Example 1 from [1].
* `example_tracking.m`: Example 2 from [1].
* `example_nonmonotonic.m`: The example from [2].

The methods are implemented in `lib`:

* `wiener_bfps()`: Bootstrap-filter-based smoother according to [1].
* `wiener_apf()`: Auxiliary (forward) filter according to [2].
* `wiener_afps()`: Auxiliary-filter-based smoother according to [2].
* `wiwner_cpfas()`: Auxiliary-filter-based conditional particle filter with
  ancestor sampling for use in particle Gibbs.

The remaining functions in `lib` are helper functions.

N.B.: This code also makes use of some functions from the [libsmc](https://github.com/rhostettler/libsmc) library.
