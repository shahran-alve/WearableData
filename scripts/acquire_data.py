#!/usr/bin/env python
"""Acquisition status, registration, and verification. Stops at permission boundaries."""
import argparse
import json

import _bootstrap  # noqa: F401

from wearabledata.acquisition import sources


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["studentlife", "depresjon", "all"], default="all")
    ap.add_argument("--register", action="store_true", help="hash raw files and write manifest")
    ap.add_argument("--verify", action="store_true", help="re-hash and compare to manifest")
    a = ap.parse_args()

    names = ["studentlife", "depresjon"] if a.dataset == "all" else [a.dataset]
    blocked = 0
    for ds in names:
        st = sources.check(ds)
        print(f"\n=== {ds} ===")
        print(f"state : {st.state}")
        print(f"reason: {st.reason}")
        print(f"source: {st.landing_page}")
        if st.state != "PRESENT":
            blocked += 1
            print("\nREQUIRED USER ACTION:\n" + st.action_required)
            continue
        if a.register:
            print("manifest:", sources.register(ds))
        if a.verify:
            print(json.dumps(sources.verify(ds), indent=2))
    if blocked:
        print(f"\n{blocked} dataset(s) blocked at the access boundary. "
              "No substitute data was generated.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
