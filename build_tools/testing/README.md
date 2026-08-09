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

Run steps default to `exit: 0`. Stdout and stderr are ignored unless checks are
declared:

```json
{
  "stdout": {
    "contains": ["first literal", {"regex": "second .+ regex"}],
    "not_contains": ["forbidden literal"]
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
      "path": "{tmp}/module.vmfb",
      "non_empty": true
    }
  ]
}
```
