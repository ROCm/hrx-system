// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/attributes.h"

#define IREE_TEST_STRINGIFY_IMPL(value) #value
#define IREE_TEST_STRINGIFY(value) IREE_TEST_STRINGIFY_IMPL(value)

#if defined(IREE_COMPILER_GCC_COMPAT) || defined(IREE_COMPILER_MSVC_COMPAT)
_Static_assert(sizeof(IREE_TEST_STRINGIFY(IREE_ATTRIBUTE_ALWAYS_INLINE)) > 1,
               "Supported compilers must provide an always-inline annotation");
_Static_assert(sizeof(IREE_TEST_STRINGIFY(IREE_ATTRIBUTE_NOINLINE)) > 1,
               "Supported compilers must provide a noinline annotation");
#endif

#include "iree/base/api.h"

static IREE_ATTRIBUTE_ALWAYS_INLINE int iree_attributes_test_add_one(
    int value) {
  return value + 1;
}

// Exercises C linkage and prefix placement alongside the public API
// annotations.
IREE_API_EXPORT IREE_MUST_USE_RESULT IREE_ATTRIBUTE_NOINLINE int
iree_attributes_test_c(int value) {
  return iree_attributes_test_add_one(value);
}
