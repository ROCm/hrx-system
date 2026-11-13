# Embedded assembly source contracts

These fixtures verify source literal admission, semantic operand/result
flattening, effect ordering, and source diagnostics for embedded assembly.
The synthetic representation contract exercises the importer boundary without
selecting a hardware backend. Target assembly lowering belongs with that target.

Related cases use `// ====` in a `.cxx-test` file. Update exact expectations
with the owning test target and `--test_arg=--update`, then run it normally.
