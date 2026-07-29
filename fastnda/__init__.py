# Copyright © 2026, Empa.
"""Public API."""

from typing import TYPE_CHECKING, Any

from fastnda.main import read, read_metadata
from fastnda.version import __version__

if TYPE_CHECKING:
    from fastnda.btsda import btsda_csv_to_parquet
    from fastnda.dicts import step_type_map

__all__ = [
    "__version__",
    "btsda_csv_to_parquet",
    "read",
    "read_metadata",
    "step_type_map",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy imports so `import fastnda` doesn't pull in polars/numpy."""
    if name == "btsda_csv_to_parquet":
        from fastnda.btsda import btsda_csv_to_parquet

        return btsda_csv_to_parquet
    if name == "step_type_map":
        from fastnda.dicts import step_type_map

        return step_type_map
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
