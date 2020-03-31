#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Nonparametric Standardized Drought Indices - pySDI main script
Author
------
    Roberto A. Real-Rangel (rrealr@iingen.unam.mx)

License
-------
    GNU General Public License
"""
# %% Import modules
from collections import OrderedDict
from datetime import datetime as dt
from pathlib2 import Path
import numpy as np
import ogr
import sys
import toml
import xarray as xr

from pysdi import drought_definition as drgt


# %% Define classes
class Configurations():
    """
    """
    def __init__(self, config_file):
        self.config_file = config_file
        config = toml.load(config_file)

        for key, value in config.items():
            setattr(self, key, value)


#%% Define functions
def load_dir(directory):
    """
    Parameters
    ----------
        directory: string
            Full path of the directory to be loaded.
    """
    if not Path(directory).exists():
        create_dir = raw_input(
            "The directory '{}' does not exist.\n"
            "Do you want to create it? [Y] Yes, [N] No. ".
            format(directory)
            )

        if create_dir.lower() == 'y':
            Path(directory).mkdir(parents=True, exist_ok=True)

        else:
            sys.exit("Cannot continue without this directory. Aborting.")

    return(Path(directory))




def check_source(raw_dataset, imp_dataset, geoextent=None):
    """Import MERRA-2 dataset.

    Parameters
    ----------
    raw_dataset : string
        Local directory where raw (monthly) MERRA-2 datasets are
        stored.
    imp_dataset : string
        Local directory where annualy aggregated datasets will be
        stored.
    """
    # TODO: Check and update the last year dataset if it is out of date.
    # OR always import any missing AND the last/current year.
    print("- Checking imported dataset in '{}'".format(imp_dataset))

    files_list = []

    for ftype in ['**/*.nc', '**/*.nc4']:
        files_list.extend(load_dir(raw_dataset).glob(ftype))

    years = sorted(set([i.stem.split('.')[2][:4] for i in files_list]))

    for year in years:
        yearfiles = list(load_dir(imp_dataset).glob(pattern='*' + year + '*'))

        if (len(yearfiles) == 0) or (year == years[-1]):
            print("  - Importing dataset of {}.".format(year))
            sources = [
                i
                for i in files_list
                if i.stem.split('.')[2][:4] == year
                ]
            raw_data = xr.open_mfdataset([str(i) for i in sources])

            if geoextent:
                xidx = np.where(
                    (raw_data['lon'] >= geoextent[0]) &
                    (raw_data['lon'] <= geoextent[1])
                    )[0]
                yidx = np.where(
                    (raw_data['lat'] >= geoextent[2]) &
                    (raw_data['lat'] <= geoextent[3])
                    )[0]
                xslice = slice(xidx.min() - 1, xidx.max() + 2)
                yslice = slice(yidx.min() - 1, yidx.max() + 2)
                raw_data = raw_data.isel(lon=xslice, lat=yslice)
                raw_data.to_netcdf(
                str(load_dir(imp_dataset) / ('M2TMNXLND_' + year + '.nc4'))
                )


def convert_units(data):
    """
    Converts the original units of MERRA-2 variables to conventional
    units (e.g., kg m-2 s-1 to mm s-1).

    Parameters
    ----------
        data: xarray.Dataset
            Dataset of the values which units are to be
            transformed.

    Returns
    -------
        xarray.Dataset
            Dataset of the transformed values.
    """
    data.time.values = data.time.values.astype('datetime64[M]')
    time_m = data.time.values.astype('datetime64[M]')
    time_s = data.time.values.astype('datetime64[s]')
    seconds = ((time_m + np.timedelta64(1, 'M') - time_s).
               astype('datetime64[s]')).astype(int)

    for var in data.var():
        try:
            if data[var].units == 'kg m-2 s-1':
                data_aux = data[var].values

                for i, val in enumerate(data_aux):
                    data_aux[i] = val * seconds[i]

                data[var].values = data_aux
                data[var].attrs['units'] = 'mm'

            elif data[var].units == 'K':
                data[var].values = data[var].values - 273.15
                data[var].attrs['units'] = 'C'

            elif data[var].units == 'kg m-2':
                data[var].attrs['units'] = 'mm'

            else:
                pass

        except AttributeError:
            pass

    return(data)


def export_nc4(dataset, output_dir='pysdi_out', prefix=None):
    """ Export a given dataset to a NetCDF-4 file (.nc4).

    Parameters:
        dataset : xarray.Dataset
        output_dir : string, optional
        prefix : string (optional; default is None)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = (
        prefix + dataset.attrs['Title'].upper() + '_'
        + dataset.attrs['TemporalRange'] + '.nc4'
        )
    dataset.to_netcdf(str(Path(output_dir) / output_file))


def progress_message(current, total, message="- Processing", units=None):
    """Issue messages of the progress of the process.

    Generates a progress bar in terminal. It works within a for loop,
    computing the progress percentage based on the current item
    number and the total length of the sequence of item to iterate.

    Parameters:
        current : integer
            The last item number computed within the for loop. This
            could be obtained using enumerate() in when calling the for
            loop.
        total : integer
            The total length of the sequence for which the for loop is
            performing the iterations.
        message : string (optional; default = "- Processing")
            A word describing the process being performed (e.g.,
            "- Computing", "- Drawing", etc.).
        units : string (optional; default = None)
            Units represented by te loops in the for block (e.g.,
            "cells", "time steps", etc.).
    """
    if units is not None:
        progress = float(current)/total
        sys.stdout.write(
            "\r    {} ({:.1f} % of {} processed)".format(
                message, progress * 100, units
                )
            )

    else:
        progress = float(current)/total
        sys.stdout.write(
            "\r    {} ({:.1f} % processed)".format(message, progress * 100)
            )

    if progress < 1:
        sys.stdout.flush()

    else:
        sys.stdout.write('\n')


def monthly_dataset(date, arrays, title='SDI'):
    """Derivate single monthly outputs from the data cube of the full
    study period.

    Parameters
    ----------
        date : xarray.DataArray
            Date that defines the month from which the data exported.
        arrays : list
            Data to export.
        title : str
            Name of the index exported.
    """
    data_vars = {
        i.attrs['DroughtFeature']: i.sel({'time': date}) for i in arrays
        }
    year = str(date).split('-')[0]
    month = str(date).split('-')[1].zfill(2)
    output_dataset = xr.Dataset(data_vars=data_vars)
    attrs = OrderedDict()
    attrs['Title'] = title
    attrs['TemporalRange'] = year + month
    attrs['SouthernmostLatitude'] = min(output_dataset.lat.values)
    attrs['NorthernmostLatitude'] = max(output_dataset.lat.values)
    attrs['WesternmostLongitude'] = min(output_dataset.lon.values)
    attrs['EasternmostLongitude'] = max(output_dataset.lon.values)
    attrs['LatitudeResolution'] = output_dataset.lat.diff(dim='lat').values[0]
    attrs['LongitudeResolution'] = output_dataset.lon.diff(dim='lon').values[0]
    attrs['SpatialCoverage'] = 'Mexico'
    attrs['History'] = (
        'Original file generated:' + dt.now().isoformat()
        )
    attrs['Contact'] = ('Roberto A. Real-Rangel (rrealr@iingen.unam.mx)')
    attrs['Institution'] = (
        'Institute of Engineering of the National Autonomous University of'
        'Mexico (II-UNAM)'
        )
    attrs['Format'] = 'NetCDF-4/HDF-5'
    attrs['VersionID'] = '1.0.0'
    output_dataset.attrs = attrs
    return(output_dataset)


# %% Apply
config_file = str(Path(__file__).parent.absolute()/'moseq_calc.toml')
config = Configurations(config_file=config_file)

# Get region extent.
source_ds = ogr.Open(utf8_path=config.general['trim_vmap'])
source_layer = source_ds.GetLayer(0)
geoextent = source_layer.GetExtent()

# Check if there is any missing dataset in the import directory.
check_source(
    raw_dataset=config.general['input_dir_raw'],
    imp_dataset=config.general['input_dir_imported'],
    geoextent=geoextent
    )

# Load the input indicators (environmental variables).
print("- Opening MERRA-2 datasets.")
indicators = xr.open_mfdataset(
    paths=(config.general['input_dir_imported'] + '**/*.nc4'),
    autoclose=True,
    parallel=False
    )
indicators = indicators.chunk(chunks={'time': -1})
indicators = convert_units(indicators)

for temp_scale in config.intensity['temp_scale']:
    for index, variable in config.intensity['sdi'].iteritems():
        # Compute drought intensity.
        drought_intensity = drgt.compute_npsdi(
            data=indicators,
            temp_scale=temp_scale,
            index=index,
            variable=variable,
            output_res=config.general['output_spatial_resolution'],
            nodata=config.general['output_nodata'],
            trim_vmap=config.general['trim_vmap'],
            interp_method=config.general['output_interp_method']
            )
        arrays_to_export = [drought_intensity]

        if (config.magnitude['compute']) and (temp_scale == 1):
            # Compute drought magnitude.
            drought_magnitude = drgt.compute_magnitude(
                intensity=drought_intensity,
                severity_threshold=config.magnitude['intensity_threshold']
                )
            arrays_to_export.append(drought_magnitude)

        # Export results to .nc4 files.
        if config.general['export_last']:
            dates_to_export = [drought_intensity.time[-1].values]

        else:
            dates_to_export = [i.values for i in drought_intensity.time]

        for d, date in enumerate(dates_to_export):
            output_dataset = monthly_dataset(
                date=date,
                arrays=arrays_to_export,
                title=(index + '-' + str(temp_scale).zfill(2))
                )
            output_dir = (
                config.general['output_dir'] + '/' + index.lower()
                + str(temp_scale).zfill(2) + '/'
                + str(date).split('-')[0]
                )
            export_nc4(
                dataset=output_dataset,
                output_dir=output_dir,
                prefix=config.general['output_fname_prefix']
                )
            progress_message(
                current=(d + 1),
                total=len(dates_to_export),
                message="- Exporting results",
                units='files'
                )

print("Process ended at {}.".format(dt.now()))
