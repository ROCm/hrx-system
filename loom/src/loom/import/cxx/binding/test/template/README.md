# Loom template family bindings

These fixtures verify family declarations, applications, and provider metadata
through C++ import. C++ template argument evaluation and specialization identity
live in `../../../symbol/test`; this package owns link-selected Loom families.

Related cases use `// ====` in a `.cxx-test` file. Update exact expectations
with the owning test target and `--test_arg=--update`, then run it normally.
