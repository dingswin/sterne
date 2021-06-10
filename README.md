# STERNE (Stars in German)
aStromeTry bayEsian infeReNcE


Code Description:
This code, written in python3, generalises on astrometryfit (https://github.com/adamdeller/astrometryfit). The generalisation enables one to infer astrometric parameters for sources sharing some identical astrometric parameters (but different on other astrometric parameters). 


Functions:
1) infer 5 parameters -- parallax, proper motion and reference position; if required, also infer longitude of ascending node and inclination angle (already realized by astrometryfit).
2) for sources sharing some identical astrometric parameters (e.g. parallax, proper motion), parameters can be inferred together (new feature).
#3) one can choose to infer parallax or distance; A distance-related Galactic prior can be used for the inference (new feature).


Requisites to use Sterne: 
1) input positions (normally measured with VLBI) in the traditional "pmpar.in" format (for pmpar).
2) initsfile (.inits) containing priors for each parameter; a priminary initsfile can be produced with priors.generate_initsfile().
3) parfile (.par) containing timing parameters (provided by PSRCAT); latest numbers beyond PSRCAT should be updated if possible.
