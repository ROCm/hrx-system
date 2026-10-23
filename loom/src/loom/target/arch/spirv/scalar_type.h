// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V semantic scalar vocabulary used by native type construction and
// target legality. These are not SPIR-V wire enums such as SpvComponentType.

#ifndef LOOM_TARGET_ARCH_SPIRV_SCALAR_TYPE_H_
#define LOOM_TARGET_ARCH_SPIRV_SCALAR_TYPE_H_

typedef enum loom_spirv_scalar_type_kind_e {
  // Unknown or uninitialized scalar type kind.
  LOOM_SPIRV_SCALAR_TYPE_KIND_UNKNOWN = 0,
  // Scalar type declared with OpTypeFloat.
  LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT = 1,
  // Scalar type declared with signed OpTypeInt.
  LOOM_SPIRV_SCALAR_TYPE_KIND_SIGNED_INT = 2,
  // Scalar type declared with unsigned OpTypeInt.
  LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT = 3,
} loom_spirv_scalar_type_kind_t;

typedef enum loom_spirv_scalar_type_e {
  // Unknown or uninitialized scalar type.
  LOOM_SPIRV_SCALAR_TYPE_UNKNOWN = 0,
  // IEEE binary16 floating-point scalar type.
  LOOM_SPIRV_SCALAR_TYPE_F16 = 1,
  // IEEE binary32 floating-point scalar type.
  LOOM_SPIRV_SCALAR_TYPE_F32 = 2,
  // IEEE binary64 floating-point scalar type.
  LOOM_SPIRV_SCALAR_TYPE_F64 = 3,
  // KHR bfloat16 scalar type, declared as OpTypeFloat 16 BFloat16KHR.
  LOOM_SPIRV_SCALAR_TYPE_BF16 = 4,
  // Signed 8-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_S8 = 5,
  // Signed 16-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_S16 = 6,
  // Signed 32-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_S32 = 7,
  // Signed 64-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_S64 = 8,
  // Unsigned 8-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_U8 = 9,
  // Unsigned 16-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_U16 = 10,
  // Unsigned 32-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_U32 = 11,
  // Unsigned 64-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_U64 = 12,
} loom_spirv_scalar_type_t;

#endif  // LOOM_TARGET_ARCH_SPIRV_SCALAR_TYPE_H_
