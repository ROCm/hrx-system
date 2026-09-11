// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/libhsa.h"

// HSA headers declare entry points without Windows import/export linkage.
// Emit exports separately so definitions retain that linkage and intentionally
// omitted optional entry points remain absent from the test library.
#if defined(_WIN32)
#define IREE_LIBHSA_TEST_EXPORT(symbol) \
  __pragma(comment(linker, "/export:" #symbol))
#else
#define IREE_LIBHSA_TEST_EXPORT(symbol) __attribute__((visibility("default")))
#endif  // _WIN32

#define IREE_LIBHSA_TEST_RETURN_hsa_status_t return HSA_STATUS_SUCCESS;
#define IREE_LIBHSA_TEST_RETURN_void return;
#define IREE_LIBHSA_TEST_RETURN_uint32_t return 0;
#define IREE_LIBHSA_TEST_RETURN_uint64_t return 0;
#define IREE_LIBHSA_TEST_RETURN_hsa_signal_value_t return 0;

#define IREE_HAL_AMDGPU_LIBHSA_PFN(trace_category, result_type, symbol, decl, \
                                   args)                                      \
  IREE_LIBHSA_TEST_EXPORT(symbol) result_type HSA_API symbol(decl) {          \
    IREE_LIBHSA_TEST_RETURN_##result_type                                     \
  }
#define IREE_HAL_AMDGPU_LIBHSA_OPTIONAL_LEAK_CHECK_DISABLED_PFN(...)
#define DECL(...) __VA_ARGS__
#include "iree/hal/drivers/amdgpu/util/libhsa_tables.h"
