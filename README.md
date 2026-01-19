<h1 align="center">
  <img src="https://github.com/user-attachments/assets/9fb435b9-ca5b-4d68-9437-2de86bef45ec" width="400" align="center" alt="aurora-biologic logo">
</h1>

<br>

[![PyPI version](https://img.shields.io/pypi/v/fastnda.svg)](https://pypi.org/project/fastnda/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/g-kimbell/fastnda/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/fastnda.svg)](https://pypi.org/project/fastnda/)
[![Checks](https://img.shields.io/github/actions/workflow/status/g-kimbell/fastnda/test.yml)](https://github.com/g-kimbell/fastnda/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/codecov/c/github/g-kimbell/fastnda)](https://app.codecov.io/gh/g-kimbell/fastnda)

Python and command-line tool to read Neware .nda and .ndax files fast.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d0f43b0d-feba-41f9-8303-26aa99844192" width="500" align="center" alt="Aurora cycler manager">
</p>
<p align="center">
  Time to convert a ~100 MB, 1.3-million-row .ndax file to .csv. Best of three runs.<br>1) Cold start from command-line interface, including module imports.<br>2) Processing time only, without UI navigation.
</p>

<br>

This project is a fork of [`NewareNDA`](https://github.com/d-cogswell/NewareNDA), and builds on top of projects [`neware_reader`](https://github.com/FTHuld/neware_reader) and [`nda-extractor`](https://github.com/thebestpatrick/nda-extractor). `fastnda` uses [`polars`](https://github.com/pola-rs/polars) and parallelization to significantly reduce processing time.


## Installation

Install from PyPI:
```shell
pip install fastnda
pip install fastnda[extras]  # If you want to write HDF5 or pandas-readable files
```


## Using with Python

Import and use `read` for both .nda and .ndax files:
```python
import fastnda

df = fastnda.read("my/neware/file.ndax")  # Returns a polars dataframe

df = df.to_pandas()  # If you prefer to work with pandas dataframes

metadata = fastnda.read_metadata("my/neware/file.ndax")  # Get metadata as a dictionary
```


## Using the command-line interface

Use help to see functions, arguments, options:
```shell
fastnda --help          # See all functions
fastnda convert --help  # See how convert function works
```

The command-line interface can perform single-file or batch-file conversion to various formats:
```shell
fastnda convert "my/file.ndax"                          # Converts file to "my/file.csv"
fastnda convert "my/file.ndax" "output/file.parquet"    # Convert file to different location and format
fastnda convert "my/file.ndax" --format=arrow --pandas  # Convert to old-pandas-compatible arrow

fastnda batch-convert "my/folder/"                      # Convert all nda and ndax files in a folder to csv
fastnda batch-convert "my/folder/" --format=h5          # Convert all files to hdf5
fastnda batch-convert "my/folder/" --recursive          # Search all subfolders
fastnda batch-convert "my/folder/" "output/folder/"     # Save all files in a different folder

fastnda print-metadata "my/file.ndax"                   # Print metadata to terminal
fastnda convert-metadata "my/file.ndax"                 # Convert metadata to my/file.json
```


## Help! My file can't be read / is converted incorrectly

Usually this is due to a hardware setting or a file type we have not seen before. Raise a GitHub issue here or on `NewareNDA`, send some test data, and we will add support.


## Notes

This package adheres closely to the outputs from Neware's BTSDA, but there are some differences:
  - Capacity and energy are one column, charge is positive and discharge is negative
  - A negative current during charge will count negatively to the capacity, in Neware it is ignored
  - In some Neware files, cycles are only counted when the step index goes backwards
  - Here, a cycle is when a charge -> discharge has been completed
  - Change this behaviour with `cycle_mode = x` or `--cycle-mode=x`, where `x` is `'chg' | 'dchg' | 'auto' | 'raw'`
  - Neware sometimes uses "DChg" and sometimes "Dchg" for discharge, here it is always "DChg"
  - Neware "Pulse Step" is here "Pulse"

Differences compared to `NewareNDA`
  - `fastnda` returns `polars` dataframes with different column names
  - There is only one capacity and one energy column
  - Time is explicitly split into step time and total time

Other benefits of using `fastnda` or `NewareNDA` over Neware's BTSDA:
  - Batch or automated file conversion is straightforward with Python or CLI
  - BTSDA drops precision depending on the units you select, e.g. exporting to V is less precise than exporting to mV
  - BTSDA can drop precision over time, e.g. after 1e6 seconds, all millisecond precision can be dropped
  - Different BTSDA versions need to be installed to open different .nda or .ndax files

Pandas compatibility
 - Old versions of `pyarrow` had an issue converting categorical columns to `pandas`, fixed in 23.0.0.
 - For compatibility with `pandas` in parquet/arrow/feather files, you can:
   - Update to `pyarrow >= 23.0.0`.
   - Convert to `pandas` first with `to_pandas()` or `--pandas`.
   - Use integer codes for the categorical columns with `raw_categories=True` or `--raw-categories`. 


## Contributions

If you have problems reading data, raise an issue on GitHub.

Code contributions are very welcome, clone the repo and use `pip install -e .[dev]` for developer dependencies.


## Acknowledgements

This project and its upstream [`NewareNDA`](https://github.com/d-cogswell/NewareNDA) are both made possible through community contributions from scientists with different ideas, equipment, and data. This fork explores larger, performance-focused changes that are difficult to land upstream incrementally. NewareNDA remains mature and actively maintained, and collaboration with the upstream project is ongoing.

This software was developed at the Laboratory of Materials for Energy Conversion at Empa, the Swiss Federal Laboratories for Materials Science and Technology, and supported by funding from the [IntelLiGent](https://heuintelligent.eu/) project from the European Union’s research and innovation program under grant agreement No. 101069765, and from the Swiss State Secretariat for Education, Research, and Innovation (SERI) under contract No. 22.001422.

<img src="https://github.com/user-attachments/assets/373d30b2-a7a4-4158-a3d8-f76e3a45a508#gh-light-mode-only" height="100" alt="IntelLiGent logo">
<img src="https://github.com/user-attachments/assets/9d003d4f-af2f-497a-8560-d228cc93177c#gh-dark-mode-only" height="100" alt="IntelLiGent logo">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/user-attachments/assets/1d32a635-703b-432c-9d42-02e07d94e9a9" height="100" alt="EU flag">&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/user-attachments/assets/cd410b39-5989-47e5-b502-594d9a8f5ae1" height="100" alt="Swiss secretariat">
