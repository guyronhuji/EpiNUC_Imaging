#!/usr/bin/env bash
# Register the notebook-stripping clean filter. Run once per clone.
#
# Git filter configuration lives in .git/config, which is NOT cloned — so every clone that intends
# to COMMIT notebooks must run this, or it will commit their outputs. Read-only clones need nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
git config filter.nbstrip.clean "python3 tools/nbstrip.py"
git config filter.nbstrip.smudge cat
git config filter.nbstrip.required true
echo "Registered filter.nbstrip — notebook outputs will be stripped from commits."
