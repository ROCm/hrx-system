// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V low boundary ABI metadata.

#ifndef LOOM_TARGET_ARCH_SPIRV_ABI_H_
#define LOOM_TARGET_ARCH_SPIRV_ABI_H_

#include "iree/base/api.h"
#include "loom/target/arch/spirv/value_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_SPIRV_ABI_ARG_VALUE_TYPES_ATTR_NAME "spirv_arg_value_types"
#define LOOM_SPIRV_ABI_RESULT_VALUE_TYPES_ATTR_NAME "spirv_result_value_types"

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_ABI_H_
