#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  printf 'usage: %s <loom-link> <loom-format> <subject> <dependency> <duplicate-a> <duplicate-b>\n' "$0" >&2
  exit 64
fi

resolve_runfile() {
  local path="$1"
  if [[ -n "${RUNFILES_DIR:-}" ]]; then
    printf '%s/%s/%s\n' "${RUNFILES_DIR}" "${TEST_WORKSPACE:-_main}" "${path}"
  else
    printf '%s\n' "${path}"
  fi
}

loom_link="$(resolve_runfile "$1")"
loom_format="$(resolve_runfile "$2")"
subject="$(resolve_runfile "$3")"
dependency="$(resolve_runfile "$4")"
duplicate_a="$(resolve_runfile "$5")"
duplicate_b="$(resolve_runfile "$6")"

subject_text="${TEST_TMPDIR}/subject.loom"
linked_bytecode="${TEST_TMPDIR}/linked.loombc"
linked_text="${TEST_TMPDIR}/linked.loom"

"${loom_format}" "${subject}" --to=text --output="${subject_text}"
grep -Fq 'func.def public @entry' "${subject_text}"
grep -Fq 'func.def @helper' "${subject_text}"
grep -Fq 'func.def public @support' "${subject_text}"
if grep -Fq 'func.decl @support' "${subject_text}"; then
  printf 'direct-source declaration was not resolved during the merge\n' >&2
  exit 1
fi
grep -Fq 'func.decl pure @dependency' "${subject_text}"
if grep -Fq 'func.def public pure @dependency' "${subject_text}"; then
  printf 'relocatable library copied a dependency definition into its module\n' >&2
  exit 1
fi

# This separate process reloads the serialized library and resolves only the
# reachable definition from the independently propagated dependency module.
"${loom_link}" "${subject}" \
  --library="${dependency}" \
  --mode=link \
  --root=@entry \
  --to=bc \
  --output="${linked_bytecode}"
"${loom_format}" "${linked_bytecode}" \
  --to=text \
  --output="${linked_text}"

if grep -Eq '^func\.decl' "${linked_text}"; then
  printf 'completed link retained an unresolved declaration\n' >&2
  exit 1
fi
grep -Fq 'func.def public retain @entry' "${linked_text}"
grep -Fq 'func.def @helper' "${linked_text}"
grep -Fq 'func.def @support' "${linked_text}"
grep -Fq 'func.def pure @dependency' "${linked_text}"

duplicate_output="${TEST_TMPDIR}/duplicate.loombc"
duplicate_diagnostics="${TEST_TMPDIR}/duplicate.stderr"
if "${loom_link}" "${duplicate_a}" "${duplicate_b}" \
  --mode=merge \
  --to=bc \
  --output="${duplicate_output}" \
  2>"${duplicate_diagnostics}"; then
  printf 'duplicate direct definitions unexpectedly merged\n' >&2
  exit 1
fi
grep -Fq '@duplicate' "${duplicate_diagnostics}"
