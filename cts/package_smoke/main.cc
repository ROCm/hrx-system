// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include <pyre_compiler_cxx.h>
#include <pyre_runtime_cxx.h>

int main() {
  pyre::compiler::compiler_ptr compiler;
  pyre::runtime::device_ptr device;
  return !compiler && !device &&
          pyre::runtime::format_status(pyre_ok_status()) == "OK"
      ? 0
      : 1;
}
