# Loom TextMate Grammars

The JSON grammars in this directory are generated from the Python Loom DSL:

```bash
python3 loom/py/loom/gen/run.py textmate
```

To try the grammar in VS Code while developing, run
`Developer: Install Extension from Location` and select
`loom/src/loom/editor/vscode`.

For an isolated extension-development window instead:

```bash
code --new-window --extensionDevelopmentPath=loom/src/loom/editor/vscode .
```

Then open a `.loom` or `.loom-test` file and use `Developer: Inspect Editor
Tokens and Scopes` to inspect the TextMate scopes under the cursor.

They are intentionally lexical. TextMate highlighting should make `.loom` and
`.loom-test` files readable before semantic tooling starts, but it must not
try to validate op assembly formats, resolve SSA values, resolve symbols, infer
types, or run loom-check. Those belong in the parser-backed document model and
future LSP.

Performance rules:

- Keep patterns line-local. Do not use `[\s\S]`, `(.|\n)`, `(?s)`, or any
  equivalent multi-line wildcard.
- Prefer several small generated per-dialect op-name patterns over one giant
  whole-language regex.
- Put `.loom-test` directive/comment patterns before the base Loom include so
  check syntax wins before generic `//` comments.
- Keep nested `begin`/`end` states shallow. Strings and diagnostic comments are
  fine; op-format parsing is not.
- Avoid backtracking-heavy regexes. Use literal alternations generated from DSL
  names and simple identifier/number patterns.
- Treat TextMate scopes as fallback labels. Semantic token precision belongs in
  `loom_document_t`/LSP.
