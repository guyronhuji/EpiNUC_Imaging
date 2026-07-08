#!/usr/bin/env python3
"""Git clean filter: strip outputs from a notebook on stdin, write it to stdout.

Registered by ``tools/setup-git-filters.sh`` and applied via ``.gitattributes``. The point is that
the *working copies* keep their rendered figures and result tables (they are the record of a run),
while the blobs git stores carry only code — this repo is public and those outputs are real
microscope images and counts.

Deliberately dependency-free (json, not nbformat) so a fresh clone can register the filter without
installing anything, and byte-stable so an unchanged notebook never shows up as dirty.
"""
import json
import sys


def strip(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        # Execution timings/ids leak run history and churn the diff for no benefit.
        meta = cell.get("metadata", {})
        for k in ("execution", "collapsed", "scrolled"):
            meta.pop(k, None)
    nb.get("metadata", {}).pop("widgets", None)  # can embed rendered image data
    if "kernelspec" in nb.get("metadata", {}):
        nb["metadata"]["kernelspec"].pop("display_name", None)  # machine-specific
    return nb


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        sys.stdout.write(raw)  # not a notebook we understand; pass through untouched
        return 0
    json.dump(strip(nb), sys.stdout, indent=1, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
