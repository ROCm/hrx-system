# Loom documentation

`loom/docs/` owns the Loom programming guide and the inputs used to assemble
its published reference. It is documentation build infrastructure, not a
shipped Loom command-line surface.

Add the optional, pinned documentation toolchain to the repository environment:

```bash
python dev.py setup --docs
```

This installs the hash-locked Python requirements and the official pinned
Doxygen binary into `.venv`; it does not depend on distribution packages.

Build the strict static site:

```bash
python dev.py docs build
```

The result is written to `build/loom-docs/site/`. Serve it with live reload
while editing:

```bash
python dev.py docs serve
```

Hand-authored pages live under `src/`. Executable guide programs live under
`examples/` so the compiler and execution-test infrastructure can validate the
same source readers see. Generated dialect reference pages and C API output are
assembled only in the build tree; generated Markdown and HTML are not checked
in. Set `LOOM_DOCS_WORK_DIR` to relocate that isolated build tree when needed.
