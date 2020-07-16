# low_earth_orbit_sampling

This code was made to investigate the effect of LEO satellite sampling on the determination of
climatological averages.

This code simulates the sampling of a sun-synchronous satellite in a circular orbit
around a spherical earth.
It simulates the ground track of the satellite, and calculates the viewing/solar geometry and geolocation
of instrument pixels.
'synthetic observations' at these time/location/viewing-solar geometry are generated according to a specified model.
These synthetic observations are then gridded onto lat/lon grids at specified averaging period.

Orbital calculations are taken from
"Handbook of Satellite Orbits: From Kepler to GPS" by Michel Capderou.

Written by Jesse Loveridge (jesserl2@illinois.edu) at University of Illinois at Urbana-Champaign.

