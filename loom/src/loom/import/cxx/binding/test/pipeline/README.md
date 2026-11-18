# Pipeline and command source contracts

These fixtures verify channel handles and payload types, strand placement,
composition, scheduling regions, and the specialization/launch boundary of
command programs. They inspect verified IR and source diagnostics without
executing pipelines or selecting devices.

Call-site operand evaluation and reassembly of staged entry parameters are
covered by the translation package’s staged launch tests.

Related cases use `// ====` in a `.cxx-test` file. Update exact expectations
with the owning test target and `--test_arg=--update`, then run it normally.
