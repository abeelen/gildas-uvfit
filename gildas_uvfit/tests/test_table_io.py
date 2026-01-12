import pytest
from astropy.io import registry
from astropy.table import Table

from ..table_io import *
from ..table_io import uvfit_table_reader


def test_table_io_registry():
    formats = registry.get_formats()
    formats.add_index("Format")
    try:
        format = formats.loc["gildas_uvfit"]
        assert format["Data class"] == "Table"
        assert format["Read"] == "Yes"
        assert format["Auto-identify"] == "Yes"
    except KeyError:
        assert False, "gildas_uvfit is not in the registry"


def test_table_io_read():
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data"
    test_file = data_path / "uvfit.fits"
    with pytest.raises(registry.IORegistryError):
        data = Table.read(test_file)

    data = Table.read(test_file, format="gildas_uvfit")
    assert data.meta["OBJECT"] == "BLANK"
    assert len(data) == 497
    for colname in ["ra_offset", "dec_offset", "flux", "fwhm"]:
        assert f"{colname}_1" in data.colnames
        assert f"e_{colname}_1" in data.colnames

    assert "flux_2" not in data.colnames
    assert "fwhm_major_1" not in data.colnames


def test_uvfit_reader_from_package_data():
    """Ensure uvfit_table_reader works on the packaged sample file."""
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data"
    test_file = data_path / "uvfit.fits"
    table = uvfit_table_reader(str(test_file))
    assert table.meta["OBJECT"] == "BLANK"
    # basic columns should be present
    assert "rms" in table.colnames
    assert any(c in table.colnames for c in ["ra_offset_1", "dec_offset_1", "flux_1", "fwhm_1"])
