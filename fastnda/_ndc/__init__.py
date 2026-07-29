# Copyright © 2026, Empa.
"""Private NDC readers used by NDAX module.

Do not use the private methods read_ndc_{type}_{number} directly.
They may change any time without warning.
Use the public "read_ndax" or "read_ndc".
"""

from .ndc import read_ndc

__all__ = ["read_ndc"]
