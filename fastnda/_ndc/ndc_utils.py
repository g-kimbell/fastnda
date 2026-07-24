"""Private functions shared by all ndc readers.

Do not use these methods directly, they may change any time without warning.
"""

import numpy as np
import polars as pl


def bytes_to_df(
    buf: bytes,
    dtype: np.dtype,
    data_start_ind: int = 132,
    record_size: int = 4096,
    file_header_records: int = 1,
    record_end_pad: int = 1,
    *,
    use_bitmask: bool = True,
    add_index: bool = False,
) -> pl.DataFrame:
    """Read bytes into a polars DataFrame.

    Args:
        buf: Bytes object containing the binary data.
        dtype: Numpy dtype describing the record structure.
        data_start_ind: Index in bytes of the start of the data in the record.
        record_size: Total size of a single record in bytes.
        file_header_records: Number of records in the file header.
        record_end_pad: Number of bytes at the end of the record that cannot contain data.
        use_bitmask: Whether to use bitmask to filter data.
        add_index: Whether to add an index column, used for filetype 1.

    Returns:
        DataFrame containing the data, dropping columns starting with '_'.

    """
    # Read entire file into 1 byte array nrecords x record_size
    num_records = len(buf) // record_size - file_header_records
    arr = np.frombuffer(
        buf,
        dtype=np.uint8,
        offset=record_size * file_header_records,
    ).reshape((num_records, record_size))
    rows_per_record = (record_size - data_start_ind - record_end_pad) // dtype.itemsize

    if use_bitmask:
        bitmask_start = 4
        bits_in_bitmask = int(np.ceil(rows_per_record / 8))
        bitmask = arr[:, bitmask_start : bitmask_start + bits_in_bitmask]
        bitmask = np.unpackbits(bitmask, bitorder="little", axis=1)[:, :rows_per_record].ravel()

    # Remove padding columns
    useful_cols = [name for name in dtype.names if not name.startswith("_")]
    dtype_no_pad = dtype[useful_cols]

    # Slice the data
    data_end_ind = data_start_ind + dtype.itemsize * rows_per_record
    sliced = arr[:, data_start_ind:data_end_ind].view(dtype_no_pad)
    columns = {name: np.ascontiguousarray(sliced[name]).ravel() for name in dtype_no_pad.names}

    df = pl.DataFrame(columns)

    if not use_bitmask:
        return df.filter(pl.col("index") != 0)
    if add_index:
        df = df.with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("index"))
    return df.filter(pl.Series(bitmask).ne(0))
