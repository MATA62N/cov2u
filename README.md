<img alt="FIDUCEO: MMD_HARM" align="right" src="http://www.fiduceo.eu/sites/default/files/FIDUCEO-logo.png">

# cov2u

Dev code for Monte Carlo estimation of parameter uncertainty from parameter covariance matrices generated by [H2020 FIDUCEO](https://fiduceo.eu). Adapted from code originally written by Jonathan Mittaz.

## Contents

* `cov2u.py` - main script to be run with Python 3.6+

The first step is to clone the latest cov2u code and step into the check out directory: 

    $ git clone https://github.com/patternizer/cov2u.git
    $ cd plot_l1c
    
### Using Standard Python 

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested in a conda virtual environment running a 64-bit version of Python 3.6+.

**cov2u** can be run from sources directly, once the following module requirements are resolved to cater for plotting with cartopy:

* `optparse`
* `xarray`
* `numpy`
* `scipy`

Run with:

    $ python cov2u.py cov.nc
           
### cov.nc

* `cov.nc` - netCDF-4 file containing the covariance matrix.

Available on request from https://github.com/patternizer

## License

The code is distributed under terms and conditions of the [MIT license](https://opensource.org/licenses/MIT).

## Contact information

* [Michael Taylor](https://patternizer.github.io)

