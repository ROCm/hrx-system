#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  printf "expected --probe, template root, and fixture path; got %d args\n" \
    "$#" >&2
  exit 1
fi

if [[ "$1" != "--probe" ]]; then
  printf "expected forwarded --probe flag, got %s\n" "$1" >&2
  exit 1
fi

if [[ "$2" != "--template-root=${PWD}" ]]; then
  printf "unexpected template root: %s (working directory: %s)\n" \
    "$2" "${PWD}" >&2
  exit 1
fi

fixture="$3"
case "${fixture}" in
  loom/build_tools/bazel/test/roundtrip.loom-test) ;;
  *)
    printf "unexpected fixture path: %s\n" "${fixture}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${fixture}" ]]; then
  printf "fixture path does not exist: %s\n" "${fixture}" >&2
  exit 1
fi
