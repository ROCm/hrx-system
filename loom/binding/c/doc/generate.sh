#!/usr/bin/env bash

set -euo pipefail

print_usage() {
  cat <<'EOF'
Usage: loom/binding/c/doc/generate.sh [--check] [--output=PATH]

Generates Loom C API documentation with Doxygen.

Options:
  --check        Run the strict documentation lint shape.
  --output=PATH  Override the output directory.
  --help         Print this help text.

Default output:
  generate mode: ${TMPDIR:-/tmp}/loom-c-api-docs
  check mode:    ${TMPDIR:-/tmp}/loom-c-api-docs-check
EOF
}

mode="generate"
output_dir=""
temp_dir="${TMPDIR:-/tmp}"
temp_dir="${temp_dir%/}"

for argument in "$@"; do
  case "${argument}" in
    --check)
      mode="check"
      ;;
    --output=*)
      output_dir="${argument#--output=}"
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: ${argument}" >&2
      print_usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${output_dir}" ]]; then
  if [[ "${mode}" == "check" ]]; then
    output_dir="${temp_dir}/loom-c-api-docs-check"
  else
    output_dir="${temp_dir}/loom-c-api-docs"
  fi
fi

if ! command -v doxygen >/dev/null 2>&1; then
  echo "doxygen not found on PATH; install doxygen to generate Loom API docs." >&2
  exit 127
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
doxyfile="${script_dir}/Doxyfile"

cd "${repo_root}"
mkdir -p "${output_dir}"
export LOOMC_DOXYGEN_OUTPUT_DIRECTORY="${output_dir}"

doxygen "${doxyfile}"

if [[ "${mode}" == "check" ]]; then
  echo "Loom documentation check passed: ${output_dir}"
else
  echo "Loom documentation generated: ${output_dir}/html/index.html"
fi
