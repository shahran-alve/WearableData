"""Source-specific readers. Each returns the same daily-panel contract:

    columns: study_id, day_index, <daily feature columns>, label

Every reader is written against the dataset's *documented* layout and asserts
that layout loudly on first contact with real files. A reader that silently
adapts to an unexpected schema is worse than one that fails, because the failure
then surfaces as a strange result three stages downstream.
"""
from .contract import PanelContractError, validate_panel  # noqa: F401
