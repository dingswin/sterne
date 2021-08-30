STERNE (stars in German) -- aStromeTry bayEsian infeReNcE

Authors: Hao Ding and Adam Deller

Availability:
This code will be able to install by "pip install sterne".

Code Description:
This code generalises on astrometryfit (https://github.com/adamdeller/astrometryfit) by Adam Deller and Scott Ransom. The generalisation enables one to infer astrometric parameters for sources sharing some identical astrometric parameters (but different on other astrometric parameters). The scope of the code is mainly VLBI astrometry of nearby pulsars, preferably the astrometry having multiple in-beam calibrators. However, there is no doubt the code can be applied to astrometry of other targets carried out beyond VLBI.

Functions:
1) infer up to 7 parameters -- parallax, proper motion, reference position, longitude of ascending node and inclination angle (already realized by astrometryfit). For the two (binary) orbital-motion parameters, the Tempo2 convention is adopted to assist comparison with pulsar timing results.
2) for sources sharing some identical astrometric parameters (e.g. parallax, proper motion), parameters can be inferred together (new feature).

Requisites to use Sterne: 
1) input positions (normally measured with VLBI) in the traditional "pmpar.in" format (for pmpar).
2) initsfile (.inits) containing priors for each parameter; a priminary initsfile can be produced with priors.generate_initsfile().
3) parfile (.par) containing timing parameters (provided by PSRCAT); latest numbers beyond PSRCAT should be updated if possible.
See sterne.simulate.py for more details.

Outputs:
The output will be provided in the "outdir" (or a folder name specified by the user) folder under the current directory. Publication-level corner plots can be made with plot.corner() that consumes posterior_samples.dat (or its like).

Usage tips:
1) be sure to clear away the outdir folder before new inference.
2) make sure the refepoch for priors match with the one used in the simulation.
