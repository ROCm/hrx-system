# Sanitizer Execution Corpus

This directory contains Loom programs that exercise sanitizer contracts through
production execution paths. Cases here are not CLI smoke tests; they are
behavioral fixtures for access, race, value, and operation diagnostics.

Each fixture should keep the program and its `check.case` expectations together
when that makes the contract easier to inspect. Prefer structured
`check.expect.event<device>` expectations over textual output matching so a new
case can identify the sanitizer mode, target, and report field that regressed.

Target-specific sanitizer fixtures are acceptable here when the behavior being
tested is inherently target/runtime specific, such as AMDGPU HAL ASAN shadow
state or workgroup-local TSAN feedback. Target-independent IR formation,
folding, and diagnostic tests belong beside the sanitizer pass or dialect that
owns that transform.
