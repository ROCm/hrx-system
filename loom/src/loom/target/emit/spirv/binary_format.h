// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V binary-format constants used by target-local emission.

#ifndef LOOM_TARGET_EMIT_SPIRV_BINARY_FORMAT_H_
#define LOOM_TARGET_EMIT_SPIRV_BINARY_FORMAT_H_

#include "loom/target/arch/spirv/isa.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_SPIRV_SCHEMA_RESERVED = 0u,
  LOOM_SPIRV_GENERATOR_LOOM = 0u,
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_BINARY_FORMAT_H_
