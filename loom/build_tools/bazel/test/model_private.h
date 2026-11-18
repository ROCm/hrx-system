// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_BUILD_TOOLS_BAZEL_TEST_MODEL_PRIVATE_H_
#define LOOM_BUILD_TOOLS_BAZEL_TEST_MODEL_PRIVATE_H_

constexpr int kPrivateBias = MODEL_LOCAL_BIAS;
static_assert(kPrivateBias == kModelBias);

#endif  // LOOM_BUILD_TOOLS_BAZEL_TEST_MODEL_PRIVATE_H_
