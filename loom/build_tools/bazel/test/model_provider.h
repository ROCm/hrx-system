// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_BUILD_TOOLS_BAZEL_TEST_MODEL_PROVIDER_H_
#define LOOM_BUILD_TOOLS_BAZEL_TEST_MODEL_PROVIDER_H_

#include <generated/model_config.h>

constexpr int kModelBias = MODEL_BIAS;
static_assert(kModelBias == MODEL_EXPECTED_BIAS);

extern "C" int model_add_bias(int value);

#endif  // LOOM_BUILD_TOOLS_BAZEL_TEST_MODEL_PROVIDER_H_
