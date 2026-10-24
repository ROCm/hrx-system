// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_BAZEL_TEST_CC_SHARED_FIXTURE_H_
#define IREE_BUILD_TOOLS_BAZEL_TEST_CC_SHARED_FIXTURE_H_

int shared_native_value(void);
int archive_stamp_value(void);
int shared_component_value(void);
int shared_middle_value(void);

// Every caller needs the native SDK even when the component code lives in a
// shared library. The native symbol is private in each shared implementation.
static inline int shared_header_value(void) { return shared_native_value(); }

#endif  // IREE_BUILD_TOOLS_BAZEL_TEST_CC_SHARED_FIXTURE_H_
