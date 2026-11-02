// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-prepared AMDGPU native kernel plan consumed by final emission.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_PLAN_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_PLAN_H_

#include "loom/target/emit/native/amdgpu/function_plan.h"
#include "loom/target/emit/native/amdgpu/kernel_record.h"

#ifdef __cplusplus
extern "C" {
#endif

// Complete target-native plan for one accepted Low emission frame.
//
// Target compilation resolves semantic acceptance, metadata, and native packet
// placement before publishing this record. Byte and listing emission consume
// it without running another compiler stage.
typedef struct loom_amdgpu_kernel_plan_t {
  // Function-local native decisions shared with callable helpers.
  loom_amdgpu_function_plan_t function;
  // Native metadata and storage contract for the kernel.
  loom_amdgpu_kernel_record_t record;
} loom_amdgpu_kernel_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_PLAN_H_
