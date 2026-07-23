"""Module to read all Neware NDC files."""

import logging
import warnings
from collections.abc import Callable

import polars as pl

from fastnda._ndc import ndc_aux, ndc_main, ndc_runinfo, ndc_step
from fastnda.utils import UnverifiedFormatWarning

logger = logging.getLogger(__name__)

# (version, filetype) keys confirmed against real data
_CONFIRMED_NDC_KEYS = frozenset(
    {
        (2, 1),
        (2, 5),
        (5, 1),
        (5, 5),
        (11, 1),
        (11, 5),
        (11, 7),
        (11, 18),
        (14, 1),
        (14, 5),
        (14, 7),
        (14, 18),
        (16, 1),
        (16, 5),
        (16, 7),
        (16, 18),
        (17, 1),
        (17, 5),
        (17, 7),
        (17, 18),
    },
)
# Map NDC (version, filetype) to handler functions
_NDC_READERS: dict[tuple[int, int], None | Callable[[bytes], pl.DataFrame]] = {
    # ndax 1
    (1, 1): ndc_main.read_ndc_main_1,
    # ndax 2
    (2, 1): ndc_main.read_ndc_main_2,
    (2, 5): ndc_aux.read_ndc_aux_2,
    # ndax 3 - probably never used
    (3, 1): ndc_main.read_ndc_main_1,
    # ndax 4 - probably never used
    (4, 1): ndc_main.read_ndc_main_2,
    (4, 5): ndc_aux.read_ndc_aux_2,
    # ndax 5
    (5, 1): ndc_main.read_ndc_main_5,
    (5, 5): ndc_aux.read_ndc_aux_5,
    # ndax 6
    (6, 1): ndc_main.read_ndc_main_6,
    (6, 5): ndc_aux.read_ndc_aux_6,
    (6, 7): ndc_step.read_ndc_step_6,
    # ndax 7
    (7, 1): ndc_main.read_ndc_main_5,
    (7, 5): ndc_aux.read_ndc_aux_5,
    # ndax 8
    (8, 1): ndc_main.read_ndc_main_14,
    (8, 5): ndc_aux.read_ndc_aux_6,
    (8, 7): ndc_step.read_ndc_step_6,
    (8, 18): ndc_runinfo.read_ndc_runinfo_1,
    # ndax 9
    (9, 1): ndc_main.read_ndc_main_11,
    (9, 5): ndc_aux.read_ndc_aux_9,
    (9, 7): ndc_step.read_ndc_step_6,
    (9, 18): ndc_runinfo.read_ndc_runinfo_2,
    # ndax 10 - probably never used
    # ndax 11
    (11, 1): ndc_main.read_ndc_main_11,
    (11, 5): ndc_aux.read_ndc_aux_11,
    (11, 7): ndc_step.read_ndc_step_11,
    (11, 18): ndc_runinfo.read_ndc_runinfo_11,
    # ndax 12
    (12, 1): ndc_main.read_ndc_main_14,
    (12, 5): ndc_aux.read_ndc_aux_6,
    (12, 7): ndc_step.read_ndc_step_6,
    (12, 18): ndc_runinfo.read_ndc_runinfo_11,
    # ndax 13
    (13, 1): ndc_main.read_ndc_main_11,
    (13, 5): ndc_aux.read_ndc_aux_9,
    (13, 7): ndc_step.read_ndc_step_6,
    (13, 18): ndc_runinfo.read_ndc_runinfo_13,
    # ndax 14
    (14, 1): ndc_main.read_ndc_main_14,
    (14, 5): ndc_aux.read_ndc_aux_6,
    (14, 7): ndc_step.read_ndc_step_6,
    (14, 18): ndc_runinfo.read_ndc_runinfo_14,
    # ndax 15 - BTS9.x, needs more investigations
    (15, 1): None,
    # ndax 16
    (16, 1): ndc_main.read_ndc_main_16,
    (16, 5): ndc_aux.read_ndc_aux_16,
    (16, 7): ndc_step.read_ndc_step_16,
    (16, 18): ndc_runinfo.read_ndc_runinfo_16,
    # ndax 17
    (17, 1): ndc_main.read_ndc_main_14,
    (17, 5): ndc_aux.read_ndc_aux_6,
    (17, 7): ndc_step.read_ndc_step_16,
    (17, 18): ndc_runinfo.read_ndc_runinfo_17,
}


def read_ndc(buf: bytes) -> pl.DataFrame:
    """Read electrochemical data from a Neware ndc binary file.

    Args:
        buf: Bytes object for the .ndc file to read
    Returns:
        DataFrame containing all records in the file

    """
    # Get ndc file version and filetype
    ndc_filetype = int(buf[0])
    ndc_version = int(buf[2])
    reader = _NDC_READERS.get((ndc_version, ndc_filetype))
    if reader is None:
        msg = f"ndc version {ndc_version} filetype {ndc_filetype} is not yet supported!"
        raise NotImplementedError(msg) from None
    if (ndc_version, ndc_filetype) not in _CONFIRMED_NDC_KEYS:
        warnings.warn(
            f"ndc version {ndc_version} filetype {ndc_filetype} has not been verified against real Neware "
            "data - results may be incorrect. If you can, please share a sample file at "
            "https://github.com/empaeconversion/fastnda/issues so we can confirm this format.",
            UnverifiedFormatWarning,
            stacklevel=2,
        )
    logger.debug("Reading ndc version %d filetype %d", ndc_version, ndc_filetype)
    return reader(buf)
