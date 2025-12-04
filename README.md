# FastNDA

Python tool to parse Neware .nda and .ndax binary files.

This project is a fork of [d-cogswell/NewareNDA](https://github.com/d-cogswell/NewareNDA), which has taken over development from [original NewareNDA project](https://github.com/Solid-Energy-Systems/NewareNDA).

This is an experimental fork refactored with a focus on speed, for those of us with enormous quantities of battery cycling data. The data parsing takes advantage of `polars` and uses vectorization where possible to give a ~10x speed improvement.

## Should I use this or NewareNDA?

FastNDA is an experimental fork. It is thoroughly tested, but the public API and dataframe columns may change before 1.0.0 release. NewareNDA is more mature and stable, and is still being actively maintained.

If you are interested in parsing your data as fast as possible and are willing to help stress test this package, use FastNDA. If you need stability, stick with NewareNDA.

## Installation

The package requires Python >3.10. Install from PyPI:
```
pip install fastnda
```

If you want to write hdf5 or pandas-readable files, install extra dependencies
```
pip install fastnda[extras]
```

## Using with Python

Import and use `read` for both .nda and .ndax

```python
import fastnda

df = fastnda.read("my/neware/file.ndax")
```
This returns a polars dataframe. If you would prefer to use pandas, you can do a zero-copy convert with:
```python
df = df.to_pandas()
```
You will need pandas and pyarrow installed for this.

> [!NOTE]
> If you want to write a to a file that uses pyarrow (e.g. parquet or feather) that can be read by both pandas and polars, you must convert to pandas first, e.g.:
> ```python
> df.to_pandas().to_parquet(filename, compression="brotli")
> ```
>
> If you write directly from polars, polars categorical/enum columns are written in a way that is very fast and space efficient, but cannot be read by pandas.
> 
> This is an issue with pyarrow/pandas and is out of my control. Here, `brotli` compression mitigates most of the space cost of storing the these columns in a pandas-friendly way.

You can also get file metadata with:
```python
metadata = fastnda.read_metadata("my/neware/file.ndax")
```
This returns a dictionary, it is comprehensive for .ndax files, but more limited in most .nda files.

## Using with command-line interface

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

## Differences between BTSDA and fastnda

This package generally adheres very closely to the outputs from BTSDA, but there are some subtle differences aside from column names:
- Capacity and energy
  - In Neware, capacity and energy can have separate columns for charge and discharge, and both can be positive
  - In fastnda, capacity and energy are one column, charge is positive and discharge is negative
  - In fastnda, a negative current during charge will count negatively to the capacity, in Neware it is ignored
- Cycle count
  - In some Neware files, cycles are only counted when the step index goes backwards, this is an inaccurate definition
  - By default in fastnda, a cycle is when a charge and discharge step have been completed (or discharge then charge)
  - The original behaviour can be accessed from fastnda, but is not generally recommended
- Status codes
  - Neware sometimes uses "DChg" and sometimes "Dchg" for discharge, fastnda always uses "DChg"

## Contributions

Contributions are very welcome.

If you have problems reading data, please raise an issue on this GitHub page.

We are always in need of test data sets, as there are many different .nda and .ndax file types, and we can only generate some with the equipment we have.

Ideally, test data is small. We need the .nda/.ndax file and may ask you for a .csv exported from BTSDA if we cannot open the file. We will only put test data in the public tests on GitHub if you agree.

Code contributions are very welcome, please clone the repo, use `pip install -e .[dev]` for dev dependencies.
