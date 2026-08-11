# Loom documentation

`loom/docs/` owns the Loom programming guide and the inputs used to assemble
its published reference. It is documentation build infrastructure, not a
shipped Loom command-line surface.

Create an isolated documentation environment from the locked dependency set:

```bash
python3 -m venv build/loom-docs/venv
build/loom-docs/venv/bin/python -m pip install \
  --require-hashes \
  -r loom/docs/requirements.lock.txt
```

Build the strict static site:

```bash
build/loom-docs/venv/bin/python loom/docs/build.py build
```

The result is written to `build/loom-docs/site/`. Serve it with live reload
while editing:

```bash
build/loom-docs/venv/bin/python loom/docs/build.py serve
```

Hand-authored pages live under `src/`. Executable guide programs live under
`examples/` so the compiler and execution-test infrastructure can validate the
same source readers see. Generated dialect reference pages and C API output are
assembled only in the build tree; generated Markdown and HTML are not checked
in.
