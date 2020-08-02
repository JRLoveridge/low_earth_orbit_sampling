"""
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
"""

import numpy as np
from collections import OrderedDict
from scipy.interpolate import interp1d
import xarray as xr
import scipy.stats as stats
import os

#---------------------------------------------------------------------------------
#------------------------------ SATELLITE/INSTRUMENT -----------------------------
#---------------------------------------------------------------------------------

class Satellite(object):

    def __init__(self, inclination, altitude, period, tAN=0.0, lamb0=0.0):
        """
        TODO
        """
        self.mu = 24*60.0/period
        self.mean_motion = self.mu*2*np.pi
        self.inclination = np.deg2rad(inclination)
        self.altitude = altitude*1000
        self.tAN = tAN
        self.R = 6371*1e3
        self.lamb0 = lamb0
        self.rho = self.R/(self.altitude+self.R)

        self._instruments = OrderedDict()

    @classmethod
    def Terra(cls):
        return cls(inclination=98.208, altitude=704.2749356577315, period=98.88)

    @classmethod
    def TerraFrozen(cls):
        return cls(inclination=98.208, altitude=704.2749356577315, period=98.88412017167381)
    
    def set_start_time(self, time_start):

        self._calculate_ground_track(t=np.linspace(0.0, time_start, 2),
                                    init_interp=False)
        #update starting time
        self.time_start = time_start

    def propagate_in_time(self, time_length, npts):
        """
        TODO
        """
        self._calculate_ground_track(t=np.linspace(self.time_start, self.time_start + time_length, npts))

        output = OrderedDict()
        for name, instrument in self._instruments.items():
            time, latitude, longitude, data = instrument()
            output[name] = {'time': time, 'latitude': latitude, 'longitude': longitude,
                           'data': data}

        #update starting time
        self.time_start += time_length

        return output

    def _calculate_ground_track(self, t, init_interp=True):
        """
        TODO
        """
        #Euler angles
        alpha2 = self.inclination
        alpha1 = self.lamb0 - (self.mean_motion/self.mu)*(t-self.tAN) #sun synchronous assumption here.
        alpha3 = self.mean_motion*(t-self.tAN)
        r = self.R+self.altitude

        #geocentric cartesian.
        X = r*(np.cos(alpha1)*np.cos(alpha3) - np.sin(alpha1)*np.sin(alpha3)*np.cos(alpha2))
        Y = r*(np.sin(alpha1)*np.cos(alpha3) + np.cos(alpha1)*np.sin(alpha3)*np.cos(alpha2))
        Z = r*(np.sin(alpha3)*np.sin(alpha2))

        #geocentric lat/lon of the sub-satellite point/ground track
        #Spherical earth so geocentric and geodetic are the same.
        psi = np.arcsin(Z/r)
        lamb = np.zeros(psi.shape)
        infill = np.arccos(X/(r*np.cos(psi)))
        lamb[np.where(Y>=0.0)] = infill[np.where(Y>=0.0)]
        lamb[np.where(Y<0.0)] = -1*infill[np.where(Y<0.0)]

        self.lamb = lamb
        self.psi = psi
        self.t = t

        if init_interp:
            self._init_interpolators()

    def _init_interpolators(self):
        """
        TODO
        """
        self.psi_interp = interp1d(self.t, self.psi, kind='linear', copy=False, bounds_error=False,
                                 assume_sorted=True)
        self.lamb_interps = []
        split_inds = np.where(np.abs(np.diff(self.lamb)) > np.pi)[0] + 1
        split_lamb = np.split(self.lamb, split_inds)
        split_time = np.split(self.t, split_inds)

        self.time_minmaxes = []
        for time,lamb in zip(split_time,split_lamb):
            if len(time) > 1:
                self.lamb_interps.append(
                interp1d(time, lamb, kind='linear', copy=False, bounds_error=True,
                                     assume_sorted=True)

                )
                self.time_minmaxes.append((time[0], time[-1]))

    def interpolate_ground_track(self, sampling_interval, descending_only=True):
        """
        TODO
        """
        new_lambs = []
        new_t = []
        for (time_min, time_max), lamb_interp in zip(self.time_minmaxes, self.lamb_interps):

            approximate = int((time_max - time_min)/sampling_interval)
            t = np.linspace(time_min + sampling_interval, time_max, approximate)
            new_lambs.append(lamb_interp(t))
            new_t.append(t)

        new_t = np.concatenate(new_t, axis=-1)
        new_psi = self.psi_interp(new_t)
        new_lamb = np.concatenate(new_lambs, axis=-1)

        if descending_only:
            diffs = np.diff(new_psi)
            cond_descend = np.where(diffs < 0.0)
            new_t = new_t[cond_descend]
            new_psi = new_psi[cond_descend]
            new_lamb = new_lamb[cond_descend]

        return new_t, new_psi, new_lamb

    def add_instruments(self, instruments):
        """
        TODO
        """
        instruments = np.atleast_1d(instruments)

        for instrument in instruments:
            instrument.set_satellite(self)
            if instrument.name is not None:
                self._instruments[instrument.name] = instrument
            else:
                name = 'instrument_{}'.format(len(list(self._instruments.keys())))
                self._instruments[name] = instrument

#-----------------------------------------------------------------------------------
class Instrument(object):

    def __init__(self, ground_swath_width, along_track_sampling_interval, across_track_samples,
                model = None, descending_only=True, name=None):
        """
        TODO
        """
        self._ground_swath_width = ground_swath_width
        self._along_track_sampling_interval = along_track_sampling_interval
        self._across_track_samples = across_track_samples
        self.set_model(model)
        self._descending_only = descending_only
        self.name = name

    @classmethod
    def MODIS(cls, model, descending_only=True):
        return cls(2330.0, 1.0/500000, 2330, name='MODIS', model=model, descending_only=descending_only)

    @classmethod
    def MISR_An(cls, model, descending_only=True):
        return cls(330.0, 1.0/500000, 330, name='MISR_An', model=model, descending_only=descending_only)

    def set_satellite(self, satellite):
        """
        TODO
        """
        self._satellite = satellite

        #set viewing geometry here too.
        #this implicitly defines a single line perspective (pinhole) camera pointing at the sub
        #satellite point. This is easily generalizable to include MISR's off-nadir cameras, for example.
        zenith_angle_range = np.linspace(-1*self.fov,self.fov,self.across_track_samples)
        signed = np.sign(zenith_angle_range)
        azimuth_angle_range = np.zeros(signed.shape)
        azimuth_angle_range[np.where(signed >=0.0)] = np.pi
        azimuth_angle_range[np.where(signed < 0.0)] = -np.pi

        self._zenith_angle_range = zenith_angle_range
        self._azimuth_angle_range = azimuth_angle_range

    def _calculate_geometries_and_geolocation(self):
        """
        TODO
        """
        t, psi, lamb = self.satellite.interpolate_ground_track(self.along_track_sampling_interval,
                                                                             self.descending_only)
        #there may be no data if, for example, descending_only is true.
        if len(t) > 0:

            psi = psi[:,np.newaxis]
            lamb = lamb[:,np.newaxis]
            time = t[:, np.newaxis]

            sensor_angles = np.repeat((self._zenith_angle_range)[np.newaxis,:], psi.shape[0], axis=0)
            sensor_azimuth = np.repeat((self._azimuth_angle_range)[np.newaxis,:], psi.shape[0], axis=0)
            time = np.repeat(time, sensor_angles.shape[1], axis=1)

            earth_angles = np.arcsin(np.sin(self._zenith_angle_range)/self.satellite.rho) - self._zenith_angle_range
            earth_angles = earth_angles[np.newaxis,:]
            vzas = earth_angles + sensor_angles

            #brng is the angle of the satellite track with the latitude circle.
            omega = 1.0/self.satellite.mu
            brng = np.arctan2(np.sqrt(np.sin(self.satellite.inclination)**2 - np.sin(psi)**2),
                            np.cos(self.satellite.inclination) - omega*np.cos(psi)**2)
            if not self.descending_only:
                import warnings
                #brng needs to be changed for the ascending node.
                warnings.warn('The swath of the ascending node is not properly modeled. See Line 236')
                #TODO treat bearing properly for ascending node.
                #brng[np.where()]

            #lat/lon calculation is the computational bottleneck with increasing pixel numbers.
            lats = np.arcsin( np.sin(psi)*np.cos(earth_angles) + np.cos(psi)*np.sin(earth_angles)*np.cos(brng))
            lons = lamb + np.arctan2(np.sin(brng)*np.sin(earth_angles)*np.cos(psi),
                                        np.cos(earth_angles) - np.sin(psi)*np.sin(lats))

            #periodic longitude condition.
            lons[np.where(lons < -np.pi)] = np.pi+(lons[np.where(lons < -np.pi)] + np.pi)
            lons[np.where(lons > np.pi)] = -np.pi + (lons[np.where(lons > np.pi)] - np.pi)

            self.latitude = lats
            self.longitude = lons
            self.sensor_zenith = vzas
            self.sensor_azimuth = sensor_azimuth
            self.time = time

            day = np.floor(self.time)
            doy = day % 365
            b = 2 * np.pi / 364.0 * (doy - 81)
            eq_of_time = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
            excess = (self.time - day)*24.0*60.0
            solar_time =(excess + 4*np.rad2deg(lons) + eq_of_time)/60
            h = np.deg2rad(15.0 * (solar_time - 12.0))

            delta = np.deg2rad(-23.45*np.cos(2*np.pi*self.time/365 + 20*np.pi/365))
            self.solar_zenith = np.arccos(np.sin(lats)*np.sin(delta) + np.cos(lats)*np.cos(delta)*np.cos(h))
            self.solar_azimuth = np.arcsin(-np.cos(delta) * np.sin(h) / np.cos(self.solar_zenith))

            return True

        else:
            return False

    def set_model(self, model):
        self._model = model

    def __call__(self):
        """
        TODO
        """
        check = self._calculate_geometries_and_geolocation()
        if check:
            data =  self._model(time=self.time, latitude=self.latitude, longitude=self.longitude,
                                 sensor_zenith=self.sensor_zenith, sensor_azimuth=self.sensor_azimuth,
                                 solar_zenith = self.solar_zenith, solar_azimuth=self.solar_azimuth)

            return self.time, self.latitude, self.longitude, data
        else:
            return [None]*4

    @property
    def satellite(self):
        return self._satellite

    @property
    def ground_swath_width(self):
        return self._ground_swath_width

    @property
    def along_track_sampling_interval(self):
        return self._along_track_sampling_interval

    @property
    def across_track_samples(self):
        return self._across_track_samples

    @property
    def fov(self):
        earth_angle = (self.ground_swath_width*1000/2.0)/self.satellite.R
        return np.arctan( np.sin(earth_angle) / (1.0/self.satellite.rho - np.cos(earth_angle)))

    @property
    def descending_only(self):
        return self._descending_only

#---------------------------------------------------------------------------------
#------------------------------ GRIDDING -------------------------------------------
#---------------------------------------------------------------------------------


class Grid(object):

    def __init__(self, lat, lon, save_period=1.0):
        """
        TODO
        """
        self._save_period = save_period
        self._lat = lat
        self._lon = lon
        self._grids = None
        self._start_time = None
        self._period_counter = None
        self._time_bins = None

        self._lat_centres = (lat[1:] + lat[:-1])/2.0
        self._lon_centres = (lon[1:] + lon[:-1])/2.0

    def set_times(self, global_start_time, global_end_time, start_time):

        self._start_time = start_time
        self._time_bins = np.arange(global_start_time, global_end_time + self._save_period, self._save_period)

    def set_save_directory(self, save_directory):
        self._save_directory = save_directory

    def _initialize_grids(self, output):
        """
        TODO
        """
        number_of_quantities = 0
        grids = OrderedDict()
        period_counter = OrderedDict()
        for name in output.keys():
            period_counter[name] = np.digitize(self._start_time, bins=self._time_bins)

        for name, instrument in output.items():
            number_of_quantities = 0
            for variable in instrument['data']:
                number_of_quantities +=1

            sum_grid = np.zeros((number_of_quantities, len(self._lat_centres) , len(self._lon_centres)), dtype=np.float32)
            count_grid = np.zeros((number_of_quantities, len(self._lat_centres) , len(self._lon_centres)), dtype=np.int)

            grids[name] = [sum_grid, count_grid]
            #period_counter[name] += 1

        self._grids = grids
        self._period_counter = period_counter

    def _new_grid(self, name):
        """
        TODO
        """
        number_of_quantities = 0
        grids = OrderedDict()

        shape = self._grids[name][0].shape

        sum_grid = np.zeros(shape, dtype=np.float32)
        count_grid = np.zeros(shape, dtype=np.int)

        self._grids[name] = [sum_grid, count_grid]
        self._period_counter[name] +=1


    def bin_and_save(self, output):
        """
        TODO
        """
        #if there is no valid data then do nothing.
        if list(output.values())[0]['time'] is not None:

            if self._grids is None:
                self._initialize_grids(output)
            i = 0
            for key, instrument in output.items():

                #get data for this day only
                digitized_time = np.digitize(instrument['time'], bins=self._time_bins)
                periods = np.arange(digitized_time.min(), digitized_time.max() + 1, 1)
                if np.all(periods > self._period_counter[key]):
                    self._save_grid(key)
                    self._new_grid(key)

                if len(periods) > 1:
                    for period in periods: #bin the data for the current save_period, save, and create a new grid for the next.
                        self._bin_data(key, instrument, digitized_time, period)
                        if period != periods[-1]:

                            self._save_grid(key)
                            self._new_grid(key)
                else:
                    #if only one save_period then accumulate the data.
                    period = periods
                    self._bin_data(key, instrument, digitized_time, period)


    def _save_grid(self, key):
        """
        TODO
        """
        sum_grid, count_grid = self._grids[key]

        to_save = xr.Dataset(
                        data_vars={
                            'gridded_sum': (['data_variables', 'latitude', 'longitude'], sum_grid),
                            'gridded_counts': (['data_variables','latitude', 'longitude' ], count_grid)
                        },
            coords = {
                'longitude': ('longitude', self._lon_centres),
                'latitude': ('latitude', self._lat_centres)
            }
        )
        name = 'climate_marble_{}_'.format(key) + '{}'.format(self._period_counter[key]).zfill(6) + '.nc'

        to_save.to_netcdf(os.path.join(self._save_directory, name), mode='w')

    def _bin_data(self, key, instrument, digitized_time, period):
        """
        TODO
        """
        time_condition = np.where(digitized_time == period)

        latitude = instrument['latitude'][time_condition]
        longitude = instrument['longitude'][time_condition]
        variable = instrument['data'][:,time_condition[0], time_condition[1]]
        returned = stats.binned_statistic_dd([latitude,longitude], variable,
                                                 bins=[np.deg2rad(self._lat), np.deg2rad(self._lon)], statistic='count')
        sum_returned = stats.binned_statistic_dd([latitude,longitude], variable,
                                                     bins=[np.deg2rad(self._lat), np.deg2rad(self._lon)],
                                                     statistic='sum')

        self._grids[key][0] += sum_returned[0]
        self._grids[key][1] += returned[0].astype(np.int)

    def save_final(self):
        """
        TODO
        """
        for key in self._grids.keys():
            self._save_grid(key)


#---------------------------------------------------------------------------------
#------------------------------ MODELS -------------------------------------------
#---------------------------------------------------------------------------------

def get_local_time(longitude, time):
    """
    TODO
    """
    import copy
    new_longitude = copy.deepcopy(longitude)


    excess = time - np.floor(time)

    solar_time =(excess*24.0*60.0 + 4*np.rad2deg(new_longitude + np.pi))/60
    angle = 1.0*np.deg2rad(15.0 * (solar_time - 12.0))

    angle[np.where(angle > np.pi)] = angle[np.where(angle > np.pi)] - 2*np.pi
    angle[np.where(angle<-1*np.pi)] = np.pi - (angle[np.where(angle < -1*np.pi)] + np.pi)

    angle[np.where(angle > 0.0)] = -angle[np.where(angle > 0.0)] + np.pi
    angle[np.where(angle < 0.0)] = angle[np.where(angle < 0.0)] + np.pi


    return angle

def cycle(amplitude=1.0,period=30.0, **kwargs):
    """
    TODO
    """
    day = kwargs['time'] % 365

    local_angle = get_local_time(kwargs['longitude'], kwargs['time'])
    return amplitude*np.expand_dims(np.cos(local_angle + day*2*np.pi/period),0)

def linear_in_time(gradient=1.0, **kwargs):
    """
    TODO UNTESTED
    """
    local_angle = get_local_time(kwargs['longitude'], kwargs['time'])
    local_time = (np.rad2deg(local_angle)/15.0 + 12.0)/24.0
    day = np.expand_dims(kwargs['time'] % 365,0)
    out = local_time + gradient*day
    
    return out

def linear_in_vza(gradient=1.0, **kwargs):
    """
    TODO
    """
    data = np.expand_dims(gradient*kwargs['sensor_zenith'],0)
    return data

def pseudo_glint_vza_only(pos=20.0,width=15.0, **kwargs):
    """
    TODO
    """
    return np.expand_dims(np.exp( - (kwargs['sensor_zenith'] - np.deg2rad(pos))**2 /(np.deg2rad(width))**2),0)

def pseudo_glint(amplitude=1.0, width=12.0, **kwargs):
    
    scat_angle = np.arccos( np.cos(kwargs['solar_zenith'])*np.cos(kwargs['sensor_zenith']) + 
                                  np.sin(kwargs['solar_zenith'])*np.sin(kwargs['sensor_zenith'])*
                                  np.cos(kwargs['sensor_azimuth'] - kwargs['solar_azimuth'])))
    
    glint_refl = amplitude*np.exp( - np.rad2deg(scat_angle)**2 /(width**2))

    return np.expand_dims(glint_refl, 0)

#---------------------------------------------------------------------------------
#------------------------------ UTILITY ------------------------------------------
#---------------------------------------------------------------------------------

def driver(start_time, stop_time, grid, satellite, save_directory,
                 orbit_pts=50000, mpi_comm=None):
    """
    TODO
    """
    global_start_time = start_time
    global_stop_time = stop_time

    if mpi_comm is not None:
        #evenly split time between processors
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()

        start_times = np.linspace(start_time, stop_time - (stop_time - start_time)/size, size)
        stop_times = np.linspace(start_time + (stop_time - start_time)/size, stop_time , size)

        stop_times -= (stop_times % grid._save_period)
        start_times -=(start_times % grid._save_period)

        start_time = start_times[rank]
        stop_time = stop_times[rank]
        
        if rank == 0:
            if not os.path.isdir(save_directory):
                os.makedirs(save_directory)

        print("I am rank '{}' of '{}' with start_time '{}' and stop_time '{}'".format(rank, size, start_time, stop_time))
    else:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
    #global times for consistent file naming across workers.
    grid.set_times(global_start_time, global_stop_time, start_time)
    grid.set_save_directory(save_directory)
    satellite.set_start_time(start_time)

    iters = 0
    while satellite.time_start < stop_time:

        time_to_propagate = 5.0/(24.0*60.0) #default to 5 minutes to accomodate very large swathes
                                            #of 1 km MODIS pixels. May be modified.
        if stop_time - satellite.time_start < time_to_propagate:
            time_to_propagate = stop_time - satellite.time_start
        output = satellite.propagate_in_time(time_to_propagate, orbit_pts)
        grid.bin_and_save(output)

        if mpi_comm is not None:
            if rank == 0:
                print(np.round(100*(satellite.time_start - start_time) / (stop_time - start_time), 2), '%')
        else:
            print(np.round(100*(satellite.time_start - start_time) / (stop_time - start_time), 2), '%')

        iters +=1
    grid.save_final() #save whatever is in memory after the loop ends.

