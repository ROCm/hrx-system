// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <hrx_compiler_cxx.h>
#include <hrx_runtime_cxx.h>

int main() {
  hrx::compiler::compiler_ptr compiler;
  hrx::runtime::device_ptr device;
  return !compiler && !device &&
          hrx::runtime::format_status(hrx_ok_status()) == "OK"
      ? 0
      : 1;
}
