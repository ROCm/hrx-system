#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  printf "usage: %s <loom-check-wrapper>\n" "$0" >&2
  exit 1
fi

wrapper="$1"
if [[ -n "${RUNFILES_DIR:-}" ]]; then
  workspace="${TEST_WORKSPACE:-_main}"
  wrapper="${RUNFILES_DIR}/${workspace}/${wrapper}"
fi

exec "${wrapper}" --probe
