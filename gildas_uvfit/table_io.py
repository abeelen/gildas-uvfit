# ~/.specutils/my_custom_loader.py
import os

from astropy.io import fits, registry
from astropy.table import QTable, Table


# Define an optional identifier. If made specific enough, this circumvents the
# need to add ``format="my-format"`` in the ``Spectrum1D.read`` call.
def identify_generic_fits(origin, *args, **kwargs):
    return isinstance(args[0], str) and os.path.splitext(args[0].lower())[1] == ".fits"


def identify_gildas_uvfit(origin, *args, **kwargs):
    is_fits = identify_generic_fits(origin, *args, **kwargs)
    with fits.open(args[0], memmap=True, **kwargs) as hdulist:
        return (
            is_fits
            and "GILDAS" in hdulist[0].header.get("ORIGIN", "")
            and "UV-FIT" in hdulist[0].header.get("CTYPE2", "")
        )


def uvfit_table_reader(file_name):
    # Read in the table by any means necessary

    data, header = fits.getdata(file_name, 0, header=True)

    # From https://www.iram.fr/IRAMFR/GILDAS/doc/html/map-html/node23.html
    # +1 extra column ?!?!?!?
    # (P1, P2, P3, Vel, A1, A2, A3)
    # (Par1, Err1, Par2, Err2, Par3, Err3, Par4, Err4, Par5, Err5, Par6, Err6, Par7, Err7) # data x number of function
    """
    POINT     Point source               : Offset R.A., Offset Dec, Flux
    E_GAUSS   Elliptic Gaussian source   : Offset R.A., Offset Dec, Flux, FWHM Axes (Major and Minor), Pos Ang
    C_GAUSS   Circular Gaussian source   : Offset R.A., Offset Dec, Flux, FWHM Axis
    C_DISK    Circular Disk              : Offset R.A., Offset Dec, Flux, Diameter
    E_DISK    Elliptical (inclined) Disk : Offset R.A., Offset Dec, Flux, Axis (Major and Minor), Pos Ang
    RING      Annulus                    : Offset R.A., Offset Dec, Flux, Diameter (Inner and Outer)
    EXPO      Exponential brightness     : Offset R.A., Offset Dec, Flux, FWHM Axis
    E_EXPO    Elliptic exponential       : Offset R.A., Offset Dec, Flux, FWHM Axes (Major and Minor), Pos Ang
    POWER-2   B = 1/r^2                  : Offset R.A., Offset Dec, Flux, FWHM Axis
    POWER-3   B = 1/r^3                  : Offset R.A., Offset Dec, Flux, FWHM Axis
    U_RING    Unresolved Annulus         : Offset R.A., Offset Dec, Flux, Radius
    E_RING    Inclined Annulus           : Offset R.A., Offset Dec, Flux, Inner, Outer, Pos Ang, Ratio
    SPERGEL   Spergel brightness profile : Offset R.A., Offset Dec, Flux, Half light radius, nu
    E_SPERGEL Elliptic Spergel profile   : Offset R.A., Offset Dec, Flux, Half light semi-Axes (Maj. and Min.), Pos Ang, nu
    """

    """
    Hence, $N=19$ when fitting a single function
    (4 generic columns + 3 columns associated to the fitted function + 6 times 2 columns per parameters)
    and $N=34$ when fitting simultaneously two functions.
    """

    nb_sim_function = (len(data) - 4) // (3 + 6 * 2)
    assert all(data[1] == nb_sim_function)

    col_names = ["rms", "nb_sim_function", "nb_total_params", "velocity"]

    param_names = []
    for i_func in range(nb_sim_function):
        assert all(data[4 + i_func * (3 + 6 * 2)] == (i_func + 1))
        # 4 generic columns
        param_names += [
            f"{item}_{i_func+1}"
            for item in ["nb_fitted_function", "code_function", "nb_fitted_parameters"]
        ]
        # 3 columns associated to the fitted function
        for item in ["ra_offset", "dec_offset", "flux"]:
            param_names += [f"{item}_{i_func+1}", f"e_{item}_{i_func+1}"]

        # 6 times 2 columns per param_names, depending on the function kind
        fct_kind = data[4 + i_func * (3 + 6 * 2) + 1]
        assert all(fct_kind == fct_kind[0])
        fct_kind = fct_kind[0]

        if fct_kind == 1:  # POINT
            items = [f"par_{ipar}" for ipar in range(4, 8)]
        elif fct_kind in [2, 8]:  # E_GAUSS E_EXPO
            items = ["fwhm_major", "fwhm_minor", "pos_angle"]
        elif fct_kind in [3, 7, 9, 10]:  # C_GAUSS  EXPO POWER-2 POWER-3
            items = [
                "fwhm",
            ] + [f"par_{ipar}" for ipar in range(5, 8)]
        elif fct_kind == 4:  # C_DISK
            items = [
                "diameter",
            ] + [f"par_{ipar}" for ipar in range(5, 8)]
        elif fct_kind == 5:  # E_DISK
            items = ["axis_major", "axis_minor", "pos_angle"] + [
                f"par_{ipar}" for ipar in range(7, 8)
            ]
        elif fct_kind == 6:  # RING
            items = ["diameter_inner", "diameter_outer"] + [
                f"par_{ipar}" for ipar in range(6, 8)
            ]
        elif fct_kind == 11:  # U_RING
            items = [
                "radius",
            ] + [f"par_{ipar}" for ipar in range(5, 7)]
        elif fct_kind == 12:  # E_RING
            items = ["inner", "outer", "pos_angle", "ratio"]
        elif fct_kind == 13:  # SPERGEL
            items = ["half_light_radius", "nu"] + [
                f"par_{ipar}" for ipar in range(6, 8)
            ]
        elif fct_kind == 14:  # E_SPERGEL
            items = ["half_light_major", "half_light_minor", "pos_angle", "nu"]
        else:
            raise ValueError(f"Unknown function kind {fct_kind}")

        for item in items:
            param_names += [f"{item}_{i_func+1}", f"e_{item}_{i_func+1}"]
    col_names += param_names

    table = QTable(
        data=data.T, names=col_names, dtype=[float] * len(col_names), meta=header
    )

    # Changing type and units
    int_col_names = ["nb_sim_function", "nb_total_params"] + [
        name
        for name in col_names
        if "nb_fitted_function" in name
        or "code_function" in name
        or "nb_fitted_parameters" in name
    ]
    for name in int_col_names:
        table[name] = table[name].astype(int)
    arcsec_col_names = [
        name
        for name in col_names
        if "ra_offset" in name
        or "dec_offset" in name
        or "fwhm" in name
        or "axis" in name
        or "diameter" in name
        or "radius" in name
        or "inner" in name
        or "outer" in name
        or "half_light" in name
    ]
    for name in arcsec_col_names:
        table[name].unit = "arcsec"
    jy_col_names = [name for name in col_names if "flux" in name]
    for name in jy_col_names:
        table[name].unit = "Jy"
    angle_col_names = [name for name in col_names if "pos_angle" in name]
    for name in angle_col_names:
        table[name].unit = "deg"

    # Removing empty columns
    table.remove_columns(
        [
            item
            for item in col_names
            if item.startswith("par") or item.startswith("e_par")
        ]
    )

    return table  # should be an instance of Table


registry.register_reader("gildas_uvfit", Table, uvfit_table_reader)
registry.register_identifier('gildas_uvfit', Table, identify_gildas_uvfit)