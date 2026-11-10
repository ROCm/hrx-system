# CLI Tests

`iree_cli_test` verifies stable behavior introduced by a public command-line
interface. An expected result here changes when the public CLI contract changes,
not when an internal compiler, linker, transform, target, testbench, runtime, or
representation changes.

## Boundary

CLI-owned behavior includes:

- a minimal successful invocation proving that a materially distinct configured
  executable loads and reaches its API;
- critical public flag combinations and conflicts;
- process exit behavior;
- process-owned help, usage, and flag diagnostics; and
- publication of an explicitly requested output file.

Semantic results belong to the API or component that establishes them. This
includes linked symbols, transformed IR, compiler diagnostics, target selection,
scenario execution, benchmark policy, structured report fields, and serialized
artifact contents. A CLI test may prove that the command published an output;
the format owner proves what that output means.

Success-only process coverage uses `iree_executable_test`. A thin wrapper that
maps IREE flags into an API normally needs one successful CLI smoke, while its
request construction and value matrix remain API tests. Common flag-library
behavior is covered once by that library instead of repeated in every tool.

Each CLI case has one declared executable, one invocation, checked-in inputs,
and assertions limited to the process boundary. Cross-tool pipelines, inline
source authoring, replay workflows, and substring checks over semantic output do
not establish a CLI contract. The manifest runner still accepts broader legacy
constructs while their ownership is being repaired; those constructs are not
part of the `iree_cli_test` contract.

## Manifest Shape

A manifest contains one or more independent cases. Each case selects one tool
and passes its public arguments directly; the runner does not use a shell.

```json
{
  "version": 1,
  "cases": [
    {
      "name": "help describes the output flag",
      "run": {
        "tool": "fixture",
        "args": ["--help"]
      },
      "stdout": {
        "contains": ["--output="]
      },
      "stderr": {
        "empty": true
      }
    }
  ]
}
```

A run defaults to `exit: 0`. An intentional public rejection uses an explicit
exit expectation and checks only the CLI-owned diagnostic:

```json
{
  "name": "rejects conflicting output destinations",
  "run": {
    "tool": "fixture",
    "args": ["--output=-", "--output_file=result.bin"]
  },
  "exit": {
    "nonzero": true
  },
  "stderr": {
    "contains": ["output destinations are mutually exclusive"]
  }
}
```

Manifest strings support `{srcdir}`, `{tmp}`, `{case}`, `{manifest}`, and
`{tool:name}` substitutions. Tool substitution is available when the build
adapter binds the tool to a single executable path. Tools launched through an
interpreter or another multi-argument command prefix can be selected by
`run.tool`, but cannot be flattened into a string substitution.

File expectations at this boundary establish publication, such as `exists` or
`non_empty`. Exact bytes, structured fields, normalization, and semantic
contents remain with the owning formatter or representation tests.

Assertion failures report the command arguments, exit status, stdout, and
stderr. Passing commands remain quiet, including commands with an expected
nonzero exit.

## Review Test

The decisive question is whether the expectation should change while the
public command line remains unchanged. A yes identifies component behavior that
happens to be visible through the executable and therefore belongs with that
component. A no identifies a genuine CLI contract or smoke.
