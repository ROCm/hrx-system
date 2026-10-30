# Execution Tests

Execution tests run command-line tools from JSON manifests and check selected
textual evidence from stdout, stderr, and generated files. The runner executes
tools directly with `subprocess.run`; it does not use a shell.

```json
{
  "version": 1,
  "cases": [
    {
      "name": "help mentions important flags",
      "run": {
        "tool": "fixture",
        "args": ["--helpish"]
      },
      "stdout": {
        "contains": ["Usage:", "--output="]
      },
      "stderr": {
        "empty": true
      }
    }
  ]
}
```

Each manifest contains one or more cases. A case can use a single `run` block or
an explicit `steps` list:

```json
{
  "cases": [
    {
      "name": "write then run",
      "steps": [
        {
          "write": {
            "path": "{tmp}/input.txt",
            "text": "hello\n"
          }
        },
        {
          "run": {
            "tool": "fixture",
            "args": ["--input={tmp}/input.txt"]
          }
        }
      ]
    }
  ]
}
```

Manifest strings support `{srcdir}`, `{tmp}`, `{case}`, `{manifest}`, and
`{tool:name}` substitutions. Tool substitution is available when the build
adapter binds the tool to a single executable path. Tools launched through an
interpreter or another multi-argument command prefix can be selected by
`run.tool`, but cannot be flattened into a string substitution.

Run steps default to `exit: 0`. Assertions on stdout and stderr are optional:

```json
{
  "stdout": {
    "contains": ["first literal", {"regex": "second .+ regex"}],
    "not_contains": ["forbidden literal"]
  }
}
```

Use `non_empty` for smoke coverage that only requires the tool to emit a stream:

```json
{
  "stdout": {
    "non_empty": true
  }
}
```

`contains` lists are ordered by default. Use `unordered` when order is not part
of the contract:

```json
{
  "stdout": {
    "contains": {
      "unordered": ["alpha", "beta"]
    }
  }
}
```

Expected failures and file checks are explicit:

```json
{
  "exit": {
    "nonzero": true
  },
  "stderr": {
    "contains": ["expected diagnostic"]
  },
  "files": [
    {
      "path": "{tmp}/artifact.bin",
      "non_empty": true
    }
  ]
}
```

A failed exit, stream, or file assertion reports the command arguments, actual
exit status, and both captured streams. Diagnostics retain the original output
even when assertions normalize it. Passing commands remain quiet, including
commands with an expected nonzero exit.

Generated artifacts can be compared byte-for-byte against another file:

```json
"files": [
  {
    "path": "{tmp}/reassembled.bin",
    "equals": "{tmp}/original.bin"
  }
]
```

File checks also support `empty`, `non_empty`, `contains`, `not_contains`, and
`normalize` with the same semantics as stream checks. A report test can select
the fields relevant to its contract without recording the complete output:

```json
"files": [
  {
    "path": "{tmp}/report.json",
    "contains": ["\"execution_count\":5"],
    "not_contains": ["\"unknown\":true"]
  }
]
```

Unrecognized file-check fields are schema errors so a misspelled assertion
cannot silently pass.

`"exists": false` checks that the command did not create a file. Content and
equality assertions require a present file and cannot be combined with it.
