// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V feature atoms and prepared feature-set views.
//
// Feature atoms are target-local units of SPIR-V extension composition. They
// own the capability/extension state needed by binary emission and the direct
// numeric bits used by target-local legality and lowering. Core Loom code
// should see only generic target records and must not depend on these tables.

#ifndef LOOM_TARGET_ARCH_SPIRV_FEATURES_H_
#define LOOM_TARGET_ARCH_SPIRV_FEATURES_H_

#include "iree/base/api.h"
#include "loom/target/arch/spirv/isa.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_spirv_feature_atom_e {
  // Unknown or uninitialized feature atom.
  LOOM_SPIRV_FEATURE_ATOM_UNKNOWN = 0,
  // Vulkan shader-module baseline.
  LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER = 1,
  // SPV_KHR_physical_storage_buffer support.
  LOOM_SPIRV_FEATURE_ATOM_PHYSICAL_STORAGE_BUFFER = 2,
  // 16-bit floating-point scalar support.
  LOOM_SPIRV_FEATURE_ATOM_FLOAT16 = 3,
  // 64-bit floating-point scalar support.
  LOOM_SPIRV_FEATURE_ATOM_FLOAT64 = 4,
  // 8-bit integer scalar support.
  LOOM_SPIRV_FEATURE_ATOM_INT8 = 5,
  // 16-bit integer scalar support.
  LOOM_SPIRV_FEATURE_ATOM_INT16 = 6,
  // 8-bit storage-buffer access support.
  LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_8BIT_ACCESS = 7,
  // 16-bit storage-buffer access support.
  LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_16BIT_ACCESS = 8,
  // SPV_NV_cooperative_vector support.
  LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_NV = 9,
  // SPV_NV_cooperative_vector training-operation support.
  LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_TRAINING_NV = 10,
  // SPV_KHR_cooperative_matrix support.
  LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR = 11,
  // SPV_KHR_bfloat16 scalar type support.
  LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_TYPE_KHR = 12,
  // SPV_KHR_bfloat16 dot-product support.
  LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_DOT_PRODUCT_KHR = 13,
  // SPV_KHR_bfloat16 cooperative-matrix support.
  LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_COOPERATIVE_MATRIX_KHR = 14,
  // 64-bit integer scalar support.
  LOOM_SPIRV_FEATURE_ATOM_INT64 = 15,
  // Vulkan memory-model Device scope support.
  LOOM_SPIRV_FEATURE_ATOM_VULKAN_MEMORY_MODEL_DEVICE_SCOPE = 16,
  // 64-bit integer atomics in storage-buffer memory.
  LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_INT64_ATOMICS = 17,
  // 64-bit integer atomics in workgroup memory.
  LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_INT64_ATOMICS = 18,
  // 16-bit floating-point basic atomics in storage-buffer memory.
  LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT16_ATOMICS = 19,
  // 16-bit floating-point basic atomics in workgroup memory.
  LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT16_ATOMICS = 20,
  // 16-bit floating-point atomic add in storage-buffer memory.
  LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD = 21,
  // 16-bit floating-point atomic add in workgroup memory.
  LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT16_ATOMIC_ADD = 22,
  // 32-bit floating-point basic atomics in storage-buffer memory.
  LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT32_ATOMICS = 23,
  // 32-bit floating-point basic atomics in workgroup memory.
  LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT32_ATOMICS = 24,
  // 32-bit floating-point atomic add in storage-buffer memory.
  LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD = 25,
  // 32-bit floating-point atomic add in workgroup memory.
  LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT32_ATOMIC_ADD = 26,
  // 64-bit floating-point basic atomics in storage-buffer memory.
  LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT64_ATOMICS = 27,
  // 64-bit floating-point basic atomics in workgroup memory.
  LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT64_ATOMICS = 28,
  // 64-bit floating-point atomic add in storage-buffer memory.
  LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD = 29,
  // 64-bit floating-point atomic add in workgroup memory.
  LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT64_ATOMIC_ADD = 30,
  // Core subgroup operations and subgroup memory scope.
  LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM = 31,
  // F32 DenormPreserve independent of other floating-point widths.
  LOOM_SPIRV_FEATURE_ATOM_FLOAT32_DENORM_PRESERVE = 32,
  // Subgroup ballot operations.
  LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_BALLOT = 33,
  // F16/F64 DenormPreserve modes selectable together without changing F32.
  LOOM_SPIRV_FEATURE_ATOM_FLOAT16_64_DENORM_PRESERVE = 34,
  // F16/F64 RoundingModeRTE modes selectable together without changing F32.
  LOOM_SPIRV_FEATURE_ATOM_FLOAT16_64_ROUNDING_RTE = 35,
  // F32 RoundingModeRTE independent of other floating-point widths.
  LOOM_SPIRV_FEATURE_ATOM_FLOAT32_ROUNDING_RTE = 36,
  // F16 SignedZeroInfNanPreserve execution-mode support.
  LOOM_SPIRV_FEATURE_ATOM_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE = 37,
  // F32 SignedZeroInfNanPreserve execution-mode support.
  LOOM_SPIRV_FEATURE_ATOM_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE = 38,
  // F64 SignedZeroInfNanPreserve execution-mode support.
  LOOM_SPIRV_FEATURE_ATOM_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE = 39,
  // Native F64-to-F16 conversion satisfies the scalar.fptrunc contract.
  LOOM_SPIRV_FEATURE_ATOM_FLOAT64_TO_FLOAT16_CONVERSION = 40,
  // Boolean votes over active subgroup invocations.
  LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_VOTE = 41,
  // Arithmetic reductions over active subgroup invocations.
  LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_ARITHMETIC = 42,
  // Arithmetic reductions over contiguous power-of-two subgroup clusters.
  LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_CLUSTERED = 43,
  // Vulkan subgroup operations admit I8/I16/I64 and F16 values.
  LOOM_SPIRV_FEATURE_ATOM_SUBGROUP_EXTENDED_TYPES = 44,
  // Subgroup indexed and XOR shuffle operations.
  LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_SHUFFLE = 45,
  // Subgroup up and down shuffle operations.
  LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE = 46,
  // SPV_KHR_integer_dot_product operations.
  LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT = 47,
  // Integer dots admit all native integer vector types.
  LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT_INPUT_ALL = 48,
  // Integer dots admit four-byte vectors.
  LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT_INPUT4X8_BIT = 49,
  // Integer dots admit four bytes packed in an i32.
  LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT_INPUT4X8_BIT_PACKED = 50,
  // Number of feature atom enum slots.
  LOOM_SPIRV_FEATURE_ATOM_COUNT = 51,
} loom_spirv_feature_atom_t;

// Bitset of loom_spirv_feature_atom_t values.
typedef uint64_t loom_spirv_feature_bits_t;

typedef enum loom_spirv_feature_bit_e {
  // Target enables Vulkan shader-module baseline.
  LOOM_SPIRV_FEATURE_VULKAN_SHADER = UINT64_C(1)
                                     << LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER,
  // Target enables SPV_KHR_physical_storage_buffer support.
  LOOM_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_PHYSICAL_STORAGE_BUFFER,
  // Target enables 16-bit floating-point scalar support.
  LOOM_SPIRV_FEATURE_FLOAT16 = UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_FLOAT16,
  // Target enables 64-bit floating-point scalar support.
  LOOM_SPIRV_FEATURE_FLOAT64 = UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_FLOAT64,
  // Target enables 8-bit integer scalar support.
  LOOM_SPIRV_FEATURE_INT8 = UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_INT8,
  // Target enables 16-bit integer scalar support.
  LOOM_SPIRV_FEATURE_INT16 = UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_INT16,
  // Target enables 8-bit storage-buffer access support.
  LOOM_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_8BIT_ACCESS,
  // Target enables 16-bit storage-buffer access support.
  LOOM_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_16BIT_ACCESS,
  // Target enables SPV_NV_cooperative_vector support.
  LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_NV,
  // Target enables SPV_NV_cooperative_vector training-operation support.
  LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_TRAINING_NV,
  // Target enables SPV_KHR_cooperative_matrix support.
  LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR,
  // Target enables SPV_KHR_bfloat16 scalar type support.
  LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_TYPE_KHR,
  // Target enables SPV_KHR_bfloat16 dot-product support.
  LOOM_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_DOT_PRODUCT_KHR,
  // Target enables SPV_KHR_bfloat16 cooperative-matrix support.
  LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_COOPERATIVE_MATRIX_KHR,
  // Target enables 64-bit integer scalar support.
  LOOM_SPIRV_FEATURE_INT64 = UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_INT64,
  // Target enables Vulkan memory-model Device scope support.
  LOOM_SPIRV_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
  // Target enables 64-bit integer atomics in storage-buffer memory.
  LOOM_SPIRV_FEATURE_STORAGE_BUFFER_INT64_ATOMICS =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_INT64_ATOMICS,
  // Target enables 64-bit integer atomics in workgroup memory.
  LOOM_SPIRV_FEATURE_WORKGROUP_INT64_ATOMICS =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_INT64_ATOMICS,
  // Target enables 16-bit floating-point basic atomics in storage-buffer
  // memory.
  LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMICS =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT16_ATOMICS,
  // Target enables 16-bit floating-point basic atomics in workgroup memory.
  LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMICS =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT16_ATOMICS,
  // Target enables 16-bit floating-point atomic add in storage-buffer memory.
  LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD,
  // Target enables 16-bit floating-point atomic add in workgroup memory.
  LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMIC_ADD =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT16_ATOMIC_ADD,
  // Target enables 32-bit floating-point basic atomics in storage-buffer
  // memory.
  LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMICS =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT32_ATOMICS,
  // Target enables 32-bit floating-point basic atomics in workgroup memory.
  LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMICS =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT32_ATOMICS,
  // Target enables 32-bit floating-point atomic add in storage-buffer memory.
  LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD,
  // Target enables 32-bit floating-point atomic add in workgroup memory.
  LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMIC_ADD =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT32_ATOMIC_ADD,
  // Target enables 64-bit floating-point basic atomics in storage-buffer
  // memory.
  LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMICS =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT64_ATOMICS,
  // Target enables 64-bit floating-point basic atomics in workgroup memory.
  LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMICS =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT64_ATOMICS,
  // Target enables 64-bit floating-point atomic add in storage-buffer memory.
  LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD,
  // Target enables 64-bit floating-point atomic add in workgroup memory.
  LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMIC_ADD =
      UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT64_ATOMIC_ADD,
} loom_spirv_feature_bit_t;

// Target enables core subgroup operations and subgroup memory scope.
// This atom occupies bit 31, which cannot be represented by a C enum on
// implementations with a 32-bit signed enum representation.
#define LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM)

// Target enables independent F32 DenormPreserve execution mode.
#define LOOM_SPIRV_FEATURE_FLOAT32_DENORM_PRESERVE \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_FLOAT32_DENORM_PRESERVE)

// Target enables subgroup ballot operations.
#define LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_BALLOT)

// Target can select F16/F64 DenormPreserve modes together without changing F32.
#define LOOM_SPIRV_FEATURE_FLOAT16_64_DENORM_PRESERVE \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_FLOAT16_64_DENORM_PRESERVE)

// Target can select F16/F64 RoundingModeRTE modes together without changing
// F32.
#define LOOM_SPIRV_FEATURE_FLOAT16_64_ROUNDING_RTE \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_FLOAT16_64_ROUNDING_RTE)

// Target enables independent F32 RoundingModeRTE execution mode.
#define LOOM_SPIRV_FEATURE_FLOAT32_ROUNDING_RTE \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_FLOAT32_ROUNDING_RTE)

// Target enables F16 SignedZeroInfNanPreserve execution mode.
#define LOOM_SPIRV_FEATURE_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE)

// Target enables F32 SignedZeroInfNanPreserve execution mode.
#define LOOM_SPIRV_FEATURE_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE)

// Target enables F64 SignedZeroInfNanPreserve execution mode.
#define LOOM_SPIRV_FEATURE_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE)

// Target native F64-to-F16 conversion satisfies the scalar.fptrunc contract.
#define LOOM_SPIRV_FEATURE_FLOAT64_TO_FLOAT16_CONVERSION \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_FLOAT64_TO_FLOAT16_CONVERSION)

// Target enables boolean votes over active subgroup invocations.
#define LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_VOTE \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_VOTE)

// Target enables arithmetic reductions over active subgroup invocations.
#define LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_ARITHMETIC \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_ARITHMETIC)

// Target enables contiguous power-of-two subgroup arithmetic clusters.
#define LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_CLUSTERED \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_CLUSTERED)

// Logical Vulkan device enables shaderSubgroupExtendedTypes.
#define LOOM_SPIRV_FEATURE_SUBGROUP_EXTENDED_TYPES \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_SUBGROUP_EXTENDED_TYPES)

// Target enables subgroup indexed and XOR shuffle operations.
#define LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_SHUFFLE)

// Target enables subgroup up and down shuffle operations.
#define LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE)

// Target enables SPV_KHR_integer_dot_product operations.
#define LOOM_SPIRV_FEATURE_INTEGER_DOT_PRODUCT \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT)

// Target admits integer dot products of all native integer vector types.
#define LOOM_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT_ALL \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT_INPUT_ALL)

// Target admits integer dot products of four-byte vectors.
#define LOOM_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT4X8_BIT \
  (UINT64_C(1) << LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT_INPUT4X8_BIT)

// Target admits integer dot products of four bytes packed in an i32.
#define LOOM_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT4X8_BIT_PACKED \
  (UINT64_C(1)                                                     \
   << LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT_INPUT4X8_BIT_PACKED)

// Feature bits known by the SPIR-V target package.
#define LOOM_SPIRV_FEATURE_KNOWN_BITS                                         \
  (LOOM_SPIRV_FEATURE_VULKAN_SHADER |                                         \
   LOOM_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER | LOOM_SPIRV_FEATURE_FLOAT16 |  \
   LOOM_SPIRV_FEATURE_FLOAT64 | LOOM_SPIRV_FEATURE_INT8 |                     \
   LOOM_SPIRV_FEATURE_INT16 | LOOM_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS | \
   LOOM_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS |                           \
   LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV |                                 \
   LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV |                        \
   LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR |                                \
   LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR |                                     \
   LOOM_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR |                              \
   LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR |                       \
   LOOM_SPIRV_FEATURE_INT64 |                                                 \
   LOOM_SPIRV_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE |                      \
   LOOM_SPIRV_FEATURE_STORAGE_BUFFER_INT64_ATOMICS |                          \
   LOOM_SPIRV_FEATURE_WORKGROUP_INT64_ATOMICS |                               \
   LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMICS |                        \
   LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMICS |                             \
   LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD |                     \
   LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMIC_ADD |                          \
   LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMICS |                        \
   LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMICS |                             \
   LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD |                     \
   LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMIC_ADD |                          \
   LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMICS |                        \
   LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMICS |                             \
   LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD |                     \
   LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMIC_ADD |                          \
   LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM |                                     \
   LOOM_SPIRV_FEATURE_FLOAT32_DENORM_PRESERVE |                               \
   LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT |                              \
   LOOM_SPIRV_FEATURE_FLOAT16_64_DENORM_PRESERVE |                            \
   LOOM_SPIRV_FEATURE_FLOAT16_64_ROUNDING_RTE |                               \
   LOOM_SPIRV_FEATURE_FLOAT32_ROUNDING_RTE |                                  \
   LOOM_SPIRV_FEATURE_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE |                  \
   LOOM_SPIRV_FEATURE_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE |                  \
   LOOM_SPIRV_FEATURE_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE |                  \
   LOOM_SPIRV_FEATURE_FLOAT64_TO_FLOAT16_CONVERSION |                         \
   LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_VOTE |                                \
   LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_ARITHMETIC |                          \
   LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_CLUSTERED |                           \
   LOOM_SPIRV_FEATURE_SUBGROUP_EXTENDED_TYPES |                               \
   LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE |                             \
   LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE |                    \
   LOOM_SPIRV_FEATURE_INTEGER_DOT_PRODUCT |                                   \
   LOOM_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT_ALL |                         \
   LOOM_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT4X8_BIT |                      \
   LOOM_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT4X8_BIT_PACKED)

// Feature bits unconditionally required by Vulkan 1.3 BDA HAL modules.
#define LOOM_SPIRV_FEATURE_MODULE_VULKAN_1_3_BDA_BASELINE \
  (LOOM_SPIRV_FEATURE_VULKAN_SHADER |                     \
   LOOM_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER | LOOM_SPIRV_FEATURE_INT64)

// Feature bits available in the built-in Vulkan 1.3 BDA target profile.
#define LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA      \
  (LOOM_SPIRV_FEATURE_MODULE_VULKAN_1_3_BDA_BASELINE | \
   LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM)

// Maximum number of OpExtension rows emitted by all modeled atoms.
#define LOOM_SPIRV_FEATURE_MAX_EXTENSION_COUNT 11
// Maximum number of OpCapability rows emitted by all modeled atoms.
#define LOOM_SPIRV_FEATURE_MAX_CAPABILITY_COUNT 35
// Maximum number of opcode rows exposed by all modeled atoms.
#define LOOM_SPIRV_FEATURE_MAX_OPCODE_COUNT 37
// Maximum number of storage-class rows exposed by all modeled atoms.
#define LOOM_SPIRV_FEATURE_MAX_STORAGE_CLASS_COUNT 1
// Maximum number of decoration rows exposed by all modeled atoms.
#define LOOM_SPIRV_FEATURE_MAX_DECORATION_COUNT 2

typedef struct loom_spirv_feature_atom_descriptor_t {
  // Stable feature atom selected by target profile bits.
  loom_spirv_feature_atom_t atom;
  // Stable feature atom name for diagnostics.
  iree_string_view_t name;
  // Transitive atom dependencies that must also be selected.
  loom_spirv_feature_bits_t required_atom_bits;
  // Minimum SPIR-V binary version required by this atom.
  uint32_t minimum_spirv_version;
  // Addressing model after this atom is selected.
  loom_spirv_addressing_model_t addressing_model;
  // Memory model after this atom is selected.
  loom_spirv_memory_model_t memory_model;
  // OpExtension string rows owned by this atom.
  const iree_string_view_t* extension_names;
  // Number of entries in |extension_names|.
  uint8_t extension_count;
  // OpCapability numeric rows owned by this atom.
  const uint32_t* capabilities;
  // Number of entries in |capabilities|.
  uint8_t capability_count;
  // Opcode rows owned by this atom.
  const uint32_t* opcodes;
  // Number of entries in |opcodes|.
  uint8_t opcode_count;
  // Storage-class rows owned by this atom.
  const uint32_t* storage_classes;
  // Number of entries in |storage_classes|.
  uint8_t storage_class_count;
  // Decoration rows owned by this atom.
  const uint32_t* decorations;
  // Number of entries in |decorations|.
  uint8_t decoration_count;
} loom_spirv_feature_atom_descriptor_t;

typedef struct loom_spirv_feature_set_t {
  // Selected feature atom bits.
  loom_spirv_feature_bits_t atom_bits;
  // Minimum SPIR-V binary version required by selected atoms.
  uint32_t minimum_spirv_version;
  // Selected module addressing model.
  loom_spirv_addressing_model_t addressing_model;
  // Selected module memory model.
  loom_spirv_memory_model_t memory_model;
  // Deterministic OpExtension rows.
  iree_string_view_t extension_names[LOOM_SPIRV_FEATURE_MAX_EXTENSION_COUNT];
  // Number of entries in |extension_names|.
  uint8_t extension_count;
  // Deterministic OpCapability rows.
  uint32_t capabilities[LOOM_SPIRV_FEATURE_MAX_CAPABILITY_COUNT];
  // Number of entries in |capabilities|.
  uint8_t capability_count;
  // Deterministic opcode rows.
  uint32_t opcodes[LOOM_SPIRV_FEATURE_MAX_OPCODE_COUNT];
  // Number of entries in |opcodes|.
  uint8_t opcode_count;
  // Deterministic storage-class rows.
  uint32_t storage_classes[LOOM_SPIRV_FEATURE_MAX_STORAGE_CLASS_COUNT];
  // Number of entries in |storage_classes|.
  uint8_t storage_class_count;
  // Deterministic decoration rows.
  uint32_t decorations[LOOM_SPIRV_FEATURE_MAX_DECORATION_COUNT];
  // Number of entries in |decorations|.
  uint8_t decoration_count;
} loom_spirv_feature_set_t;

// Returns the direct feature bit for |atom|, or zero for unknown atoms.
loom_spirv_feature_bits_t loom_spirv_feature_atom_bit(
    loom_spirv_feature_atom_t atom);

// Returns the descriptor for |atom|, or NULL when |atom| is unknown.
const loom_spirv_feature_atom_descriptor_t* loom_spirv_feature_atom_descriptor(
    loom_spirv_feature_atom_t atom);

// Returns the stable diagnostic spelling for |atom|.
iree_string_view_t loom_spirv_feature_atom_name(loom_spirv_feature_atom_t atom);

// Returns the feature atoms currently modeled by this target package.
loom_spirv_feature_bits_t loom_spirv_known_feature_bits(void);

// Includes the generated transitive dependencies of every selected atom.
// Unknown bits are preserved for validation by
// |loom_spirv_feature_set_prepare|.
loom_spirv_feature_bits_t loom_spirv_feature_bits_with_dependencies(
    loom_spirv_feature_bits_t atom_bits);

// Returns true when |feature_set| contains |atom|.
bool loom_spirv_feature_set_has_atom(
    const loom_spirv_feature_set_t* feature_set,
    loom_spirv_feature_atom_t atom);

// Prepares a deterministic feature-set view from selected target feature bits.
iree_status_t loom_spirv_feature_set_prepare(
    iree_string_view_t target_name,
    loom_spirv_feature_bits_t requested_atom_bits,
    loom_spirv_feature_set_t* out_feature_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_FEATURES_H_
