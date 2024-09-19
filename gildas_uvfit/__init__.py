from .table_io import * 
from .version import version as __version__

# Everything here is registered to the reader of astropy.table.Table or specutils.Spectrum1D
# Then you can be explicit to control what ends up in the namespace,
__all__ = []
