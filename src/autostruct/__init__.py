"""AutoStruct — sample-driven binary format structure inference.

Public entry points:

    from autostruct import infer_samples, StructModel

``infer_samples`` runs the full pipeline (load → entropy → stats → field
detection → model) and returns a :class:`~autostruct.model.StructModel`.
"""

from .model import Field, Region, StructModel, Validation
from .pipeline import InferConfig, infer_paths, infer_samples

__version__ = "0.1.0"

__all__ = [
    "Field",
    "Region",
    "StructModel",
    "Validation",
    "InferConfig",
    "infer_samples",
    "infer_paths",
    "__version__",
]
