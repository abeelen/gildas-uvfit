# ~/.specutils/my_custom_loader.py
import numpy as np

from astropy.nddata import StdDevUncertainty
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS

from specutils.io.registers import data_loader
from specutils import Spectrum1D

from .table_io import identify_gildas_uvfit


@data_loader(
    label="GILDAS UV-FIT", identifier=identify_gildas_uvfit, extensions=["fits"]
)
def gildas_uvfit(file_name, i_func=1, **kwargs):
    """[WIP] Read a GILDAS UV-FIT fits file with fixed position (and size) to return a Spectrum1D object.

    Parameters
    ----------
    file_name: str
        The path to the FITS file.
    i_func: int
        The index of the function to extract. Default is 1.
    """

    data = Table.read(file_name, format="gildas_uvfit")

    # Check that the position is not fitted (error is 0)
    assert all(
        data[f"e_ra_offset_{i_func}"] == 0
    ), "Size and position should not be fitted"
    assert all(
        data[f"e_dec_offset_{i_func}"] == 0
    ), "Size and position should not be fitted"
    # Check that the size is not fitted (error is 0)
    items = [
        name
        for name in data.colnames
        if name.endswith(f"_{i_func}")
        and (
            "e_fwhm" in name
            or "e_diameter" in name
            or "e_axis" in name
            or "e_radius" in name
            or "e_inner" in name
            or "e_outer" in name
            or "e_half_light" in name
            or "e_pos_angle" in name
        )
    ]
    for item in items:
        assert all(data[item] == 0), "Size and position should not be fitted"

    # retrieve additionnal information
    meta = {}
    items = [
        name
        for name in data.colnames
        if name.endswith(f"_{i_func}")
        and (
            name.startswith("fwhm")
            or name.startswith("diameter")
            or name.startswith("axis")
            or name.startswith("radius")
            or name.startswith("inner")
            or name.startswith("outer")
            or name.startswith("half_light")
            or name.startswith("pos_angle")
        )
    ]
    for item in items:
        meta[item] = data[item][0] * data[item].unit

    # Beware not very precise here
    phase_center = SkyCoord(data.meta["ra"], data.meta["dec"], unit="deg")
    coord_offset = SkyCoord(
        data[f"ra_offset_{i_func}"][0],
        data[f"dec_offset_{i_func}"][0],
        unit="arcsec",
        frame=phase_center.skyoffset_frame(),
    )  # .transform_to('icrs')
    coord = coord_offset.transform_to("icrs")

    meta.update(
        {"coord": coord, "phase_center": phase_center, "coord_offset": coord_offset}
    )

    flux = data[f"flux_{i_func}"].to("mJy")
    eflux = data[f"e_flux_{i_func}"].to("mJy")
    w = WCS(data.meta).sub([1])
    freqs = w.wcs_pix2world(np.arange(len(data)), 0)[0]
    # Check calculations...
    # ref_freq = hdu[0].header['RESTFREQ'] / 1e6
    # v_rad = ((ref_freq-freqs)/ref_freq * cst.c).to(u.km/u.s)
    # good at +-1.5 km/s

    return Spectrum1D(
        spectral_axis=freqs * u.MHz,
        flux=flux,
        uncertainty=StdDevUncertainty(eflux, unit="mJy"),
        meta=meta,
        velocity_convention="radio",
    )
