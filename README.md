# STERNE (Stars in German)
aStromeTry bayEsian infeReNcE


Code Description:
This code, written in python3, generalises on astrometryfit (https://github.com/adamdeller/astrometryfit). The generalisation enables one to 1) infer astrometric parameters for sources sharing some identical astrometric parameters (but different on other astrometric parameters); 2) use input positions with 3-parameter uncertainties (major-axis uncertainty, minor-axis uncertainty and position angle); 3) infer distance straight away, provided a distance-related Galactic prior.


Functions:
1) infer 5 parameters -- parallax, proper motion and reference position; if required, also infer longitude of ascending node and inclination angle (already realized by astrometryfit).
2) for sources sharing some identical astrometric parameters (e.g. parallax, proper motion), parameters can be inferred together (new feature).
3) one can choose to infer parallax or distance; A distance-related Galactic prior can be used for the inference (new feature).


Requisites to use Sterne: 
1) input positions (normally measured with VLBI) in the traditional "pmpar.in" format (for pmpar) or in an alternative format providing the angled uncertainties (with an extra beam position angle info)
2) prior for each parameter
3) timing parameters (provided by PSRCAT); latest numbers beyond PSRCAT should be updated if possible.
