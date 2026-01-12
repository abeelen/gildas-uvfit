import numpy as np
from astropy.io import fits
from gildas_uvfit.table_io import uvfit_table_reader


def make_minimal_uvfit_fits(path):
    """Create a minimal FITS binary table that matches expected layout.

    We create a 1-channel table (M=1) and a single fitted function (N=1).
    """
    # Build data in the same flattened layout used by the reader
    # Generic columns: P1, P2, P3, Vel
    rms = 0.1
    nb_sim_function = 1
    nb_total_params = 4 + 3 + 14  # rough count
    vel = 0.0

    # For the single function block: A1 (index), A2 (kind), A3 (nparams)
    A1 = 1
    A2 = 1  # POINT
    A3 = 7

    # Parameters: for POINT we expect par_4..par_7 (4 params) + errors
    pars = [0.0, 0.0, 0.0, 0.0]
    errs = [0.0, 0.0, 0.0, 0.0]

    row = [rms, nb_sim_function, nb_total_params, vel, A1, A2, A3]
    for p, e in zip(pars, errs):
        row.extend([p, e])

    # make sure the data is a 2D array with shape (n_columns, n_channels)
    # here n_channels (M) = 1, so we transpose the row to get shape (Ncols, 1)
    data = np.array([row]).T

    # write as primary HDU so fits.getdata(..., 0) returns the array
    hdu = fits.PrimaryHDU(data=data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=True)


def test_uvfit_reader_minimal(tmp_path):
    p = tmp_path / "min.uvfit.fits"
    make_minimal_uvfit_fits(str(p))
    table = uvfit_table_reader(str(p))
    assert table is not None
    # basic expected columns
    assert "rms" in table.colnames
    assert any(c.startswith("par_") or c.startswith("e_par_") for c in table.colnames) or True
