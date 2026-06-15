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

enum {
  // Feature atom does not impose an addressing model.
  LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED = UINT32_MAX,
  // Feature atom does not impose a memory model.
  LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED = UINT32_MAX,
};

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
  // Number of feature atom enum slots.
  LOOM_SPIRV_FEATURE_ATOM_COUNT = 16,
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

  // Feature bits known by the SPIR-V target package.
  LOOM_SPIRV_FEATURE_KNOWN_BITS =
      LOOM_SPIRV_FEATURE_VULKAN_SHADER |
      LOOM_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER | LOOM_SPIRV_FEATURE_FLOAT16 |
      LOOM_SPIRV_FEATURE_FLOAT64 | LOOM_SPIRV_FEATURE_INT8 |
      LOOM_SPIRV_FEATURE_INT16 | LOOM_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS |
      LOOM_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS |
      LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV |
      LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV |
      LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR |
      LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR |
      LOOM_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR |
      LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR |
      LOOM_SPIRV_FEATURE_INT64,
  // Feature bits selected by the built-in Vulkan 1.3 BDA profile.
  LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA =
      LOOM_SPIRV_FEATURE_VULKAN_SHADER |
      LOOM_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER | LOOM_SPIRV_FEATURE_INT64,
} loom_spirv_feature_bit_t;

// Maximum number of OpExtension rows emitted by all modeled atoms.
#define LOOM_SPIRV_FEATURE_MAX_EXTENSION_COUNT 7
// Maximum number of OpCapability rows emitted by all modeled atoms.
#define LOOM_SPIRV_FEATURE_MAX_CAPABILITY_COUNT 16
// Maximum number of opcode rows exposed by all modeled atoms.
#define LOOM_SPIRV_FEATURE_MAX_OPCODE_COUNT 12
// Maximum number of storage-class rows exposed by all modeled atoms.
#define LOOM_SPIRV_FEATURE_MAX_STORAGE_CLASS_COUNT 1
// Maximum number of decoration rows exposed by all modeled atoms.
#define LOOM_SPIRV_FEATURE_MAX_DECORATION_COUNT 2

typedef struct loom_spirv_feature_atom_descriptor_t {
  // Stable feature atom selected by target profile bits.
  loom_spirv_feature_atom_t atom;
  // Stable feature atom name for diagnostics.
  iree_string_view_t name;
  // Atom bits that must also be selected.
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
