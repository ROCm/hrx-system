// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Default Passthrough Interceptor
//
// This interceptor simply returns the real function table, causing all
// HIP calls to pass through directly to the backend library.
//
// Use this as a template for creating custom interceptors.
//===----------------------------------------------------------------------===//

#include "passthrough/hip_function_table.h"

// Initialize the interceptor.
// Receives the table of real HIP functions.
// Returns the function table to use (NULL = use real table directly).
__attribute__((visibility("default")))
hip_function_table_t* hip_interceptor_init(hip_function_table_t* real_functions) {
  // Return NULL to use the real functions directly.
  // If you want to intercept calls, create your own function table
  // that wraps the real functions and return that instead.
  return NULL;
}

// Optional shutdown function.
__attribute__((visibility("default")))
void hip_interceptor_shutdown(void) {
  // Nothing to clean up for the simple passthrough.
}
