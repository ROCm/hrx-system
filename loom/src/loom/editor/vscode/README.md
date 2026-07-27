# Loom VS Code Extension

This is a development-only VS Code extension wrapper around the generated
TextMate grammars in `../textmate/`.

To install it into the current VS Code session, run:

1. `Developer: Install Extension from Location`
2. Select `loom/src/loom/editor/vscode`

For an isolated extension-development window instead, run from the repository
root:

```bash
code --new-window --extensionDevelopmentPath=loom/src/loom/editor/vscode .
```

Then open a `.loom` or `.loom-test` file and use `Developer: Inspect Editor
Tokens and Scopes` to inspect the TextMate scopes under the cursor.
