"""Type aliases mirroring lightning.pytorch.utilities.types where useful."""
from __future__ import annotations

from typing import Any, Mapping, Union

STEP_OUTPUT = Union[Any, Mapping[str, Any], None]
EPOCH_OUTPUT = list[STEP_OUTPUT]
