# ~/.specutils/my_custom_loader.py
import logging
import os
from enum import IntEnum, unique
from typing import Any, Dict, List

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


# Mapping of GILDAS function kinds to parameter name templates.

""" 
    POINT       Point source                      : None
    E_GAUSS   Elliptic Gaussian source   : FWHM Axes (Major and Minor), Pos Ang
    C_GAUSS   Circular Gaussian source   : FWHM Axis
    C_DISK    Circular Disk              : Diameter
    E_DISK    Elliptical (inclined) Disk : Axis (Major and Minor), Pos Ang
    RING      Annulus                    : Diameter (Inner and Outer)
    EXPO      Exponential brightness     : FWHM Axis
    E_EXPO    Elliptic exponential       : FWHM Axes (Major and Minor), Pos Ang
    POWER-2   B = 1/r^2                  : FWHM Axis
    POWER-3   B = 1/r^3                  : FWHM Axis
    U_RING    Unresolved Annulus         : Radius
    E_RING    Inclined Annulus           : Inner, Outer, Pos Ang, Ratio
    SPERGEL   Spergel brightness profile : Half light radius, nu
    E_SPERGEL Elliptic Spergel profile   : Half light semi-Axes (Maj. and Min.), Pos Ang, nu
"""


@unique
class FunctionKind(IntEnum):
    POINT = 1
    E_GAUSS = 2
    C_GAUSS = 3
    C_DISK = 4
    E_DISK = 5
    RING = 6
    EXPO = 7
    E_EXPO = 8
    POWER_2 = 9
    POWER_3 = 10
    U_RING = 11
    E_RING = 12
    SPERGEL = 13
    E_SPERGEL = 14


# Base parameter lists (after the X and Y offset, flux common to all models).
BASE_PARAM_ITEMS = {
    FunctionKind.POINT: [],
    FunctionKind.E_GAUSS: ["fwhm_major", "fwhm_minor", "pos_angle"],
    FunctionKind.E_EXPO: ["fwhm_major", "fwhm_minor", "pos_angle"],
    FunctionKind.C_GAUSS: ["fwhm"],
    FunctionKind.EXPO: ["fwhm"],
    FunctionKind.POWER_2: ["fwhm"],
    FunctionKind.POWER_3: ["fwhm"],
    FunctionKind.C_DISK: ["diameter"],
    FunctionKind.E_DISK: ["axis_major", "axis_minor", "pos_angle"],
    FunctionKind.RING: ["diameter_inner", "diameter_outer"],
    FunctionKind.U_RING: ["radius"],
    FunctionKind.E_RING: ["inner", "outer", "pos_angle", "ratio"],
    FunctionKind.SPERGEL: ["half_light_radius", "nu"],
    FunctionKind.E_SPERGEL: ["half_light_major", "half_light_minor", "pos_angle", "nu"],
}


def generate_generic_pars(start: int, end: int = 7) -> List[str]:
    """Generate generic parameter names "par_{n}" from start to end (inclusive).

    Parameters
    ----------
    start : int
        Starting parameter index (inclusive).
    end : int, optional
        Ending parameter index (inclusive), by default 7.

    Returns
    -------
    list of str
        List of parameter names, e.g. ["par_4", "par_5", ...].

    Examples
    --------
    >>> generate_generic_pars(4)
    ["par_4", "par_5", "par_6", "par_7"]
    """
    if start > end:
        return []
    return [f"par_{i}" for i in range(start, end + 1)]


def build_kind_param_items(base_map: Dict[FunctionKind, List[str]]) -> Dict[FunctionKind, List[str]]:
    """Build the full mapping of function kinds to parameter name lists.

    For each function kind, the named base parameters occupy the earliest
    parameter slots starting at index 4; generic ``par_n`` names are
    appended to fill slots up to ``par_7`` as needed.

    Parameters
    ----------
    base_map : dict
        Mapping from :class:`FunctionKind` to a list of base parameter names.

    Returns
    -------
    dict
        Mapping from :class:`FunctionKind` to the full parameter name list
        including any appended generic ``par_n`` names.
    """
    result: Dict[FunctionKind, List[str]] = {}
    for kind, base in base_map.items():
        start = 4 + len(base)
        items = base + generate_generic_pars(start)
        result[kind] = items
    return result


# Build final mapping at module import time.
KIND_PARAM_ITEMS = build_kind_param_items(BASE_PARAM_ITEMS)


def parse_function_kind(raw: Any) -> "FunctionKind":
    """Parse a raw function-kind value into :class:`FunctionKind`.

    Parameters
    ----------
    raw : object
        Raw value extracted from the FITS table (usually an array element).

    Returns
    -------
    FunctionKind
        The corresponding enum member.

    Raises
    ------
    ValueError
        If the raw value cannot be converted to an int or does not map to a
        known :class:`FunctionKind`.
    """
    try:
        kind_value = int(raw)
    except Exception:
        raise ValueError(f"Invalid function kind value: {raw}")
    try:
        return FunctionKind(kind_value)
    except ValueError:
        raise ValueError(f"Unknown function kind {kind_value}")


def build_column_names(data, n_functions: int) -> List[str]:
    """Build column names for the UV-fit table from binary table data.

    Parameters
    ----------
    data : array-like
        Raw FITS table data array as returned by ``fits.getdata``.
    n_functions : int
        Number of simultaneous fitted functions.

    Returns
    -------
    list of str
        List of column names in order matching the data layout.
    """
    n_generic_columns = 4
    n_associated_columns = 3
    n_pararameters_per_function = 7 * 2
    func_offset = n_generic_columns
    func_stride = n_associated_columns + n_pararameters_per_function

    col_names = ["rms", "nb_sim_function", "nb_total_params", "velocity"]
    param_names: List[str] = []

    for i_func in range(n_functions):
        if not (data[func_offset + i_func * func_stride] == (i_func + 1)).all():
            raise ValueError(f"Unexpected function index for function {i_func+1}")

        # 4 generic columns for each function group
        param_names += [
            f"{item}_{i_func+1}" for item in ["nb_fitted_function", "code_function", "nb_fitted_parameters"]
        ]

        # 3 columns associated to the fitted function
        for item in ["ra_offset", "dec_offset", "flux"]:
            param_names += [f"{item}_{i_func+1}", f"e_{item}_{i_func+1}"]

        # parameter-specific columns depending on function kind
        raw_kind = data[func_offset + i_func * func_stride + 1]
        if not (raw_kind == raw_kind[0]).all():
            raise ValueError(f"Inconsistent function-kind entries for function {i_func+1}")
        kind = parse_function_kind(raw_kind[0])
        items = KIND_PARAM_ITEMS[kind]

        for item in items:
            param_names += [f"{item}_{i_func+1}", f"e_{item}_{i_func+1}"]

    col_names += param_names
    return col_names


def apply_units(table: QTable, col_names: List[str]) -> None:
    """Set units on columns of the provided table in-place.

    Parameters
    ----------
    table : QTable
        Table whose columns will be annotated with units.
    col_names : list of str
        List of column names previously used to build the table.
    """
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
        if name in table.colnames:
            table[name].unit = "arcsec"
    jy_col_names = [name for name in col_names if "flux" in name]
    for name in jy_col_names:
        if name in table.colnames:
            table[name].unit = "Jy"
    angle_col_names = [name for name in col_names if "pos_angle" in name]
    for name in angle_col_names:
        if name in table.colnames:
            table[name].unit = "deg"


def uvfit_table_reader(file_name: str) -> Table:
    """Read a GILDAS UV-FIT FITS file and return an Astropy ``Table``.

    The function parses the FITS binary table provided by GILDAS UV-FIT and
    constructs a :class:`astropy.table.QTable` with appropriate column
    names and units. Column naming follows the GILDAS UV-FIT convention
    where generic columns are followed by per-function parameter groups.

    Parameters
    ----------
    file_name : str
        Path to the FITS file to read.

    Returns
    -------
    astropy.table.Table
        A table containing the parsed fit results with units attached where
        appropriate.

    Notes
    -----
    The underlying binary table encodes up to 7 parameters per fitted
    function (each with associated errors), plus a few generic columns
    (rms, number of functions, total params, velocity). This reader
    reconstructs column names and assigns units to columns like offsets
    and angles.
    """
    # Read in the table by any means necessary

    data, header = fits.getdata(file_name, 0, header=True)

    # From https://www.iram.fr/IRAMFR/GILDAS/doc/html/map-html/node23.html
    # +1 extra column ?!?!?!?
    # (P1, P2, P3, Vel, A1, A2, A3)
    # (Par1, Err1, Par2, Err2, Par3, Err3, Par4, Err4, Par5, Err5, Par6, Err6, Par7, Err7) # data x number of function

    """
    The results are stored in a GDF table which can read with the COLUMN command. 
    The table is organized as a $M \times N$ matrix. $M$ is the number of channels in the input table. 
    The organization along the other axis is as follows P1 P2 P3 Vel A1 A2 A3 Par1 Err1 Par2 Err2 ... A1 A2 A3 Par1 Err1 ... where
    
    P1
        RMS of the fitting process. 
    P2
        Number of simultaneously fitted functions. 
    P3
        Total number of fitted parameters. 
    Vel
        Velocity of the ith channel (i lies between 1 and M). 
    A1
        Number of the fitted function (1 for the first function, 2 for the second function, ...). 
    A2
        Code of the function kind (POINT = 1, E_GAUSS = 2, ...). 
    A3
        Number of fitted parameters in the current function. 
    Par1
        Value of the first fitted parameters. 
    Err1
        Uncertainty on the value of the first fitted parameters. 

    Hence, $N=19$ when fitting a single function (4 generic columns + 3 columns associated to the fitted function + 6 times 2 columns per parameters)
    and $N=34$ when fitting simultaneously two functions. 
    This data format as the use of the task itself is not very convenient. 
    We thus recommend to use the procedures and/or the associated widgets. 
    """
    # Actually they are 7 parameters per function, each with its error, so 14 columns per function

    n_generic_columns = 4
    n_associated_columns = 3
    n_pararameters_per_function = 7 * 2

    func_stride = n_associated_columns + n_pararameters_per_function

    if (len(data) - n_generic_columns) % func_stride != 0:
        raise ValueError("FITS table length incompatible with expected layout")

    n_functions = (len(data) - n_generic_columns) // func_stride

    # Build column names using helper (validates per-function indices/kinds)
    col_names = build_column_names(data, n_functions)

    table = QTable(data=data.T, names=col_names, dtype=[float] * len(col_names), meta=header)

    # Changing type of integer-like columns
    int_col_names = ["nb_sim_function", "nb_total_params"] + [
        name
        for name in col_names
        if "nb_fitted_function" in name or "code_function" in name or "nb_fitted_parameters" in name
    ]
    for name in int_col_names:
        try:
            table[name] = table[name].astype(int)
        except Exception:
            logging.warning("Failed to convert column %s to int", name)

    # Apply units in a helper
    apply_units(table, col_names)

    # Remove generic par columns if present
    to_remove = [c for c in col_names if (c.startswith("par_") or c.startswith("e_par_")) and c in table.colnames]
    if to_remove:
        try:
            table.remove_columns(to_remove)
        except Exception:
            logging.warning("Failed to remove parameter columns: %s", to_remove)

    return table


registry.register_reader("gildas_uvfit", Table, uvfit_table_reader)
registry.register_identifier("gildas_uvfit", Table, identify_gildas_uvfit)
