#!/usr/bin/env python
"""Report the Python version, library versions, and the global seed."""
import json
import sys

import _bootstrap  # noqa: F401

from wearabledata.utils.manifest import environment_record
from wearabledata.utils.seeds import seed_everything


def main() -> int:
    rec = environment_record()
    rec["global_seed"] = seed_everything("global")
    rec["python_ok"] = sys.version_info[:2] in [(3, 11), (3, 12)]
    print(json.dumps(rec, indent=2))
    return 0 if rec["python_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
