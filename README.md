[![PyPI version](https://img.shields.io/pypi/v/fastnda.svg)](https://pypi.org/project/fastnda/)
[![Python Versions](https://img.shields.io/pypi/pyversions/fastnda.svg)](https://pypi.org/project/fastnda/)
[![Build](https://github.com/g-kimbell/fastnda/actions/workflows/test.yml/badge.svg)](https://github.com/g-kimbell/fastnda/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/g-kimbell/fastnda/graph/badge.svg?token=BB3FA6IKER)](https://codecov.io/gh/g-kimbell/fastnda)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/g-kimbell/fastnda/blob/main/LICENSE)


# _FastNDA_

Python tool to read Neware .nda and .ndax binary files fast.

This project is a fork of [`NewareNDA`](https://github.com/d-cogswell/NewareNDA), and builds on top of other projects [`neware_reader`](https://github.com/FTHuld/neware_reader) and [`nda-extractor`](https://github.com/thebestpatrick/nda-extractor).

This fork explores larger, performance-focused changes that are difficult to land upstream incrementally. NewareNDA remains mature and actively maintained, and collaboration with the upstream project is ongoing.

FastNDA uses `polars`, parallelization, and vectorized buffer reading to significantly reduce processing time.

<p align="center">
  <img src="https://github.com/user-attachments/assets/59abf7f2-c5b9-4e48-b1cc-4a4a133ee168" width="500" align="center" alt="Aurora cycler manager">
</p>
<p align="center">
  Time to convert a ~100 MB, 1.3-million-row .ndax file to .csv. Best of three runs.<br>1) Cold start from command-line interface, including module imports.<br>2) Processing time only, without UI navigation.
</p>

## Installation

The package requires Python >=3.10. Install from PyPI:
```
pip install fastnda
```

If you want to write HDF5 or `pandas`-readable files, install extra dependencies
```
pip install fastnda[extras]
```

## Using with Python

Import and use `read` for both .nda and .ndax

```python
import fastnda

df = fastnda.read("my/neware/file.ndax")
```
This returns a polars dataframe. If you would prefer to use pandas, you can do a zero-copy conversion with:
```python
df = df.to_pandas()
```
You will need pandas and pyarrow installed for this.

You can also get file metadata as a dictionary with:
```python
metadata = fastnda.read_metadata("my/neware/file.ndax")
```

## Using the command-line interface

The command-line interface can:

- Convert single .nda or .ndax files
- Batch convert folders containing .nda or .ndax files (optionally recursively)
- Convert to different file formats (csv, parquet, hdf5, arrow)
- Print or save .nda or .ndax metadata as JSON

To see all functions, use the help:
```bash
fastnda --help
```

You can also use help within a function:
```bash
fastnda convert --help
```

> [!NOTE]
> If you want to write files that use arrow (e.g. parquet/arrow/feather) that can be read by both `pandas` and `polars`, you must convert to `pandas` first.
> In Python:
> ```python
> df.to_pandas().to_parquet(filename, compression="brotli")
> ```
> 
> In the CLI, pass the `--pandas` or `-p` flag:
> ```bash
> fastnda convert "my/neware/file.ndax" --format=parquet --pandas
> ```
>
> If you write directly from `polars`, categorical columns are written in a way that cannot be read by pandas.
> This is an issue with pyarrow/pandas, not FastNDA.

## Differences between BTSDA and FastNDA

This package generally adheres very closely to the outputs from BTSDA, but there are some subtle differences aside from column names:
- Capacity and energy
  - In Neware, capacity and energy can have separate columns for charge and discharge, and both can be positive
  - In FastNDA, capacity and energy are one column, charge is positive and discharge is negative
  - In FastNDA, a negative current during charge will count negatively to the capacity, in Neware it is ignored
- Cycle count
  - In some Neware files, cycles are only counted when the step index goes backwards
  - By default in FastNDA, a cycle is when a charge and discharge step have been completed (or discharge then charge)
  - The original behaviour can be accessed from FastNDA by setting `cycle_mode = "raw"`
- Step types
  - Neware sometimes uses "DChg" and sometimes "Dchg" for discharge, FastNDA always uses "DChg"
  - Neware "Pulse Step" is here "Pulse"

Besides speed, there are other benefits of using FastNDA or NewareNDA over BTSDA:
  - Batch or automated file conversion is straightforward with Python or CLI
  - BTSDA drops precision depending on the units you select, e.g. exporting to V is less precise than exporting to mV
  - BTSDA drops time precision over time, e.g. after 1e6 seconds, all millisecond precision can be dropped
  - BTSDA can also drop capacity/energy precision over time
  - Different BTSDA versions need to be installed to open different .nda or .ndax files

## Differences between FastNDA and NewareNDA

- `fastnda` returns `polars` dataframes, `NewareNDA` returns `pandas` dataframes
- Column names have changed
- There is only one capacity and one energy column
- Time is explicitly split into step time and total time

## Contributions

Contributions are very welcome.

If you have problems reading data, please raise an issue on this GitHub page.

We are always in need of test data sets, as there are many different .nda and .ndax file types, and we can only generate some with the equipment we have.

Ideally, test data is small. We need the .nda/.ndax file and may ask you for a .csv exported from BTSDA if we cannot open the file. We will only put test data in the public tests on GitHub if you agree.

Code contributions are very welcome, please clone the repo, use `pip install -e .[dev]` for dev dependencies.
