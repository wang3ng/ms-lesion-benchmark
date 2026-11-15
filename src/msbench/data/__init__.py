"""Dataset converters into the common case layout (see common.py)."""

from . import isbi2015, msseg2016, mslesseg
from .common import CHANNELS, CaseSpec, ConversionError, build_index, convert_case

CONVERTERS = {
    msseg2016.DATASET: msseg2016.discover,
    isbi2015.DATASET: isbi2015.discover,
    mslesseg.DATASET: mslesseg.discover,
}

__all__ = ["CHANNELS", "CONVERTERS", "CaseSpec", "ConversionError", "build_index", "convert_case"]
