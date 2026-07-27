# Installed Tests

HRX can install a relocatable CTest tree and the artifacts needed to run it
outside the build directory. The installed suite is enabled by
`HRX_INSTALL_TESTS` and is installed with the component named by
`HRX_INSTALL_TESTS_COMPONENT`, which defaults to `HrxTestsDist`.

```bash
cmake --install build/hrx-runtime \
  --prefix build/hrx-runtime/install \
  --component HrxTestsDist
ctest --test-dir build/hrx-runtime/install/share/hrx-system/tests \
  --output-on-failure
```

The installed CTest tree defaults to creating per-test temporary directories
under `<prefix>/share/hrx-system/tests/tmp`. Set `HRX_TEST_TMPDIR` to redirect
those directories to a writable external location:

```bash
HRX_TEST_TMPDIR=/tmp/hrx-system-tests \
  ctest --test-dir <prefix>/share/hrx-system/tests --output-on-failure
```

Use normal CTest filters to run a subset:

```bash
ctest --test-dir <prefix>/share/hrx-system/tests \
  --output-on-failure -R libhrx/cts
```
