from modelling import *
import os

if __name__ == '__main__':

    #comment out if no MPI
    import mpi4py.MPI as MPI
    comm = MPI.COMM_WORLD

    Terra = Satellite.Terra()

    #wrap one of the model functions with any necessary parameters.
    #kwargs will be passed from the satellite for all of the lat/lon/time/solar_zenith/viewing_zenith etc
    #information that may be necessary to evaluate the model.
    def model(**kwargs):
        return pseudo_glint(pos=20.0,width=15.0, **kwargs)

    grid = Grid(np.linspace(-90, 90, 1801), np.linspace(-180.0,180.0, 3601),
                                           save_period=1.0)

    #The pre-made MISR/MODIS have extremely high temporal sampling to simulate high resolution measurements
    #as such they are quite intensive in both memory and processing. Such a large number of simulated pixels
    #is only necessary for the highest of resolution climate marble grids (0.05 degrees)
    #instruments = Instrument(330.0, 1.0/50000, 100, model=model, descending_only=True)
    instruments=[Instrument.MODIS(model=model, descending_only=True)]
    Terra.add_instruments(instruments)

    driver(0.0, 365, grid, Terra, '/data/keeling/a/jesserl2/c/MODIS_1year_pseudo_glint', mpi_comm=comm) #set to None if no mpi
