// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Volatile writes keep constructors from being folded into the data
// initializer, preserving the runtime ordering with the linker-installed
// DT_INIT function.
__attribute__((visibility("default"))) volatile int initialization_order = 0;

// The linker installs this function as DT_INIT.
void module_initialize(void) { initialization_order = 1; }

__attribute__((constructor(101))) static void initialize_second(void) {
  initialization_order = initialization_order * 10 + 2;
}

__attribute__((constructor(102))) static void initialize_third(void) {
  initialization_order = initialization_order * 10 + 3;
}
