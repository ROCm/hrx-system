// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/arithmetic.h"

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/lower_rules.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/contracts/arithmetic_lower_rules.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

#define LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT 3u

#define MIX_F32 LOOM_AMDGPU_FMA_MIX_SOURCE_F32
#define MIX_F16LO LOOM_AMDGPU_FMA_MIX_SOURCE_F16LO
#define MIX_F16HI LOOM_AMDGPU_FMA_MIX_SOURCE_F16HI
#define FMA_MIX_F32(a, b, c) \
  LOOM_AMDGPU_DESCRIPTOR_REF_V_FMA_MIX_F32_##a##_##b##_##c
#define FMA_MIX_F32_SRC2_LIT(a, b) \
  LOOM_AMDGPU_DESCRIPTOR_REF_V_FMA_MIX_F32_##a##_##b##_F32_SRC2_LIT
#define FMA_MIXLO_F16(a, b, c) \
  LOOM_AMDGPU_DESCRIPTOR_REF_V_FMA_MIXLO_F16_##a##_##b##_##c
#define FMA_MIXHI_F16(a, b, c) \
  LOOM_AMDGPU_DESCRIPTOR_REF_V_FMA_MIXHI_F16_##a##_##b##_##c
#define MAD_MIX_F32(a, b, c) \
  LOOM_AMDGPU_DESCRIPTOR_REF_V_MAD_MIX_F32_##a##_##b##_##c
#define MAD_MIXLO_F16(a, b, c) \
  LOOM_AMDGPU_DESCRIPTOR_REF_V_MAD_MIXLO_F16_##a##_##b##_##c
#define MAD_MIXHI_F16(a, b, c) \
  LOOM_AMDGPU_DESCRIPTOR_REF_V_MAD_MIXHI_F16_##a##_##b##_##c

#define FMA_MIXLO_F16_ALL_F32 FMA_MIXLO_F16(F32, F32, F32)
#define FMA_MIXHI_F16_ALL_F32 FMA_MIXHI_F16(F32, F32, F32)
#define MAD_MIXLO_F16_ALL_F32 MAD_MIXLO_F16(F32, F32, F32)
#define MAD_MIXHI_F16_ALL_F32 MAD_MIXHI_F16(F32, F32, F32)

#define LOOM_AMDGPU_FMA_MIX_DESCRIPTOR_REF_TABLE(REF, ALL_F32_REF) \
  {                                                                \
      [MIX_F32] =                                                  \
          {                                                        \
              [MIX_F32] =                                          \
                  {                                                \
                      [MIX_F32] = ALL_F32_REF,                     \
                      [MIX_F16LO] = REF(F32, F32, F16LO),          \
                      [MIX_F16HI] = REF(F32, F32, F16HI),          \
                  },                                               \
              [MIX_F16LO] =                                        \
                  {                                                \
                      [MIX_F32] = REF(F32, F16LO, F32),            \
                      [MIX_F16LO] = REF(F32, F16LO, F16LO),        \
                      [MIX_F16HI] = REF(F32, F16LO, F16HI),        \
                  },                                               \
              [MIX_F16HI] =                                        \
                  {                                                \
                      [MIX_F32] = REF(F32, F16HI, F32),            \
                      [MIX_F16LO] = REF(F32, F16HI, F16LO),        \
                      [MIX_F16HI] = REF(F32, F16HI, F16HI),        \
                  },                                               \
          },                                                       \
      [MIX_F16LO] =                                                \
          {                                                        \
              [MIX_F32] =                                          \
                  {                                                \
                      [MIX_F32] = REF(F16LO, F32, F32),            \
                      [MIX_F16LO] = REF(F16LO, F32, F16LO),        \
                      [MIX_F16HI] = REF(F16LO, F32, F16HI),        \
                  },                                               \
              [MIX_F16LO] =                                        \
                  {                                                \
                      [MIX_F32] = REF(F16LO, F16LO, F32),          \
                      [MIX_F16LO] = REF(F16LO, F16LO, F16LO),      \
                      [MIX_F16HI] = REF(F16LO, F16LO, F16HI),      \
                  },                                               \
              [MIX_F16HI] =                                        \
                  {                                                \
                      [MIX_F32] = REF(F16LO, F16HI, F32),          \
                      [MIX_F16LO] = REF(F16LO, F16HI, F16LO),      \
                      [MIX_F16HI] = REF(F16LO, F16HI, F16HI),      \
                  },                                               \
          },                                                       \
      [MIX_F16HI] =                                                \
          {                                                        \
              [MIX_F32] =                                          \
                  {                                                \
                      [MIX_F32] = REF(F16HI, F32, F32),            \
                      [MIX_F16LO] = REF(F16HI, F32, F16LO),        \
                      [MIX_F16HI] = REF(F16HI, F32, F16HI),        \
                  },                                               \
              [MIX_F16LO] =                                        \
                  {                                                \
                      [MIX_F32] = REF(F16HI, F16LO, F32),          \
                      [MIX_F16LO] = REF(F16HI, F16LO, F16LO),      \
                      [MIX_F16HI] = REF(F16HI, F16LO, F16HI),      \
                  },                                               \
              [MIX_F16HI] =                                        \
                  {                                                \
                      [MIX_F32] = REF(F16HI, F16HI, F32),          \
                      [MIX_F16LO] = REF(F16HI, F16HI, F16LO),      \
                      [MIX_F16HI] = REF(F16HI, F16HI, F16HI),      \
                  },                                               \
          },                                                       \
  }

static const loom_amdgpu_descriptor_ref_t
    kFmaMixF32DescriptorRefs[LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                            [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                            [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT] =
                                LOOM_AMDGPU_FMA_MIX_DESCRIPTOR_REF_TABLE(
                                    FMA_MIX_F32,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_NONE);

static const loom_amdgpu_descriptor_ref_t kFmaMixF32Src2LiteralDescriptorRefs
    [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
    [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT] = {
        [MIX_F32] =
            {
                [MIX_F32] = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                [MIX_F16LO] = FMA_MIX_F32_SRC2_LIT(F32, F16LO),
                [MIX_F16HI] = FMA_MIX_F32_SRC2_LIT(F32, F16HI),
            },
        [MIX_F16LO] =
            {
                [MIX_F32] = FMA_MIX_F32_SRC2_LIT(F16LO, F32),
                [MIX_F16LO] = FMA_MIX_F32_SRC2_LIT(F16LO, F16LO),
                [MIX_F16HI] = FMA_MIX_F32_SRC2_LIT(F16LO, F16HI),
            },
        [MIX_F16HI] =
            {
                [MIX_F32] = FMA_MIX_F32_SRC2_LIT(F16HI, F32),
                [MIX_F16LO] = FMA_MIX_F32_SRC2_LIT(F16HI, F16LO),
                [MIX_F16HI] = FMA_MIX_F32_SRC2_LIT(F16HI, F16HI),
            },
};

static const loom_amdgpu_descriptor_ref_t
    kMadMixF32DescriptorRefs[LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                            [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                            [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT] =
                                LOOM_AMDGPU_FMA_MIX_DESCRIPTOR_REF_TABLE(
                                    MAD_MIX_F32,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_NONE);

static const loom_amdgpu_descriptor_ref_t
    kFmaMixloF16DescriptorRefs[LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                              [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                              [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT] =
                                  LOOM_AMDGPU_FMA_MIX_DESCRIPTOR_REF_TABLE(
                                      FMA_MIXLO_F16, FMA_MIXLO_F16_ALL_F32);

static const loom_amdgpu_descriptor_ref_t
    kFmaMixhiF16DescriptorRefs[LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                              [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                              [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT] =
                                  LOOM_AMDGPU_FMA_MIX_DESCRIPTOR_REF_TABLE(
                                      FMA_MIXHI_F16, FMA_MIXHI_F16_ALL_F32);

static const loom_amdgpu_descriptor_ref_t
    kMadMixloF16DescriptorRefs[LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                              [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                              [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT] =
                                  LOOM_AMDGPU_FMA_MIX_DESCRIPTOR_REF_TABLE(
                                      MAD_MIXLO_F16, MAD_MIXLO_F16_ALL_F32);

static const loom_amdgpu_descriptor_ref_t
    kMadMixhiF16DescriptorRefs[LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                              [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
                              [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT] =
                                  LOOM_AMDGPU_FMA_MIX_DESCRIPTOR_REF_TABLE(
                                      MAD_MIXHI_F16, MAD_MIXHI_F16_ALL_F32);

typedef const loom_amdgpu_descriptor_ref_t
    loom_amdgpu_fma_mix_descriptor_ref_cube_t
        [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
        [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT]
        [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT];

#undef LOOM_AMDGPU_FMA_MIX_DESCRIPTOR_REF_TABLE
#undef MAD_MIXHI_F16_ALL_F32
#undef MAD_MIXLO_F16_ALL_F32
#undef FMA_MIXHI_F16_ALL_F32
#undef FMA_MIXLO_F16_ALL_F32
#undef MAD_MIXHI_F16
#undef MAD_MIXLO_F16
#undef MAD_MIX_F32
#undef FMA_MIXHI_F16
#undef FMA_MIXLO_F16
#undef FMA_MIX_F32_SRC2_LIT
#undef FMA_MIX_F32
#undef MIX_F16HI
#undef MIX_F16LO
#undef MIX_F32

static bool loom_amdgpu_type_is_f16(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F16;
}

static bool loom_amdgpu_type_is_even_packed_f16_vector(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = 0;
  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_F16 ||
      !loom_amdgpu_type_packed_16bit_float_storage(type, &payload_bit_count,
                                                   &register_count)) {
    return false;
  }
  if (payload_bit_count == 0 || (payload_bit_count % 32u) != 0 ||
      register_count == 0 ||
      register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }
  *out_register_count = register_count;
  return true;
}

static bool loom_amdgpu_type_is_even_packed_i16_vector(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = 0;
  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_I16 ||
      !loom_amdgpu_type_packed_integer_storage(type, &payload_bit_count,
                                               &register_count)) {
    return false;
  }
  if (payload_bit_count == 0 || (payload_bit_count % 32u) != 0 ||
      register_count == 0 ||
      register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }
  *out_register_count = register_count;
  return true;
}

static bool loom_amdgpu_type_is_even_packed_f32_vector(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = 0;
  const uint32_t register_count = loom_amdgpu_vector_f32_register_count(type);
  if (register_count == 0 || (register_count % 2u) != 0 ||
      register_count / 2u > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }
  *out_register_count = register_count;
  return true;
}

static bool loom_amdgpu_fma_mix_source_is_f16(
    loom_amdgpu_fma_mix_source_kind_t source_kind) {
  return source_kind == LOOM_AMDGPU_FMA_MIX_SOURCE_F16LO ||
         source_kind == LOOM_AMDGPU_FMA_MIX_SOURCE_F16HI;
}

static bool loom_amdgpu_fma_mix_source_ref_is_valid(
    loom_amdgpu_fma_mix_source_kind_t source_kind) {
  return (uint32_t)source_kind < LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT;
}

static const loom_op_t* loom_amdgpu_source_defining_op(
    const loom_module_t* module, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
}

static bool loom_amdgpu_select_fma_mix_packed_extract_source(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_id_t* out_source,
    loom_amdgpu_fma_mix_source_kind_t* out_source_kind,
    uint32_t* out_source_register_offset) {
  const loom_op_t* extract_op =
      loom_amdgpu_source_defining_op(module, value_id);
  if (extract_op == NULL || !loom_vector_extract_isa(extract_op) ||
      loom_vector_extract_result(extract_op) != value_id) {
    return false;
  }
  const loom_attribute_t static_indices =
      loom_vector_extract_static_indices(extract_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      loom_vector_extract_indices(extract_op).count != 0) {
    return false;
  }
  const int64_t lane_index = static_indices.i64_array[0];
  if (lane_index < 0 || lane_index == INT64_MIN || lane_index > UINT32_MAX) {
    return false;
  }

  const loom_value_id_t source = loom_vector_extract_source(extract_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  if (!loom_type_is_vector(source_type) ||
      loom_type_element_type(source_type) != LOOM_SCALAR_TYPE_F16) {
    return false;
  }
  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(
          source_type, &payload_bit_count, &register_count)) {
    return false;
  }
  const uint32_t lane_count = payload_bit_count / 16u;
  const uint32_t source_register_offset = (uint32_t)lane_index / 2u;
  if ((uint32_t)lane_index >= lane_count ||
      source_register_offset >= register_count) {
    return false;
  }

  *out_source = source;
  *out_source_kind = ((uint32_t)lane_index & 1u)
                         ? LOOM_AMDGPU_FMA_MIX_SOURCE_F16HI
                         : LOOM_AMDGPU_FMA_MIX_SOURCE_F16LO;
  *out_source_register_offset = source_register_offset;
  return true;
}

bool loom_amdgpu_select_fma_mix_source(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_id_t* out_source,
    loom_amdgpu_fma_mix_source_kind_t* out_source_kind,
    uint32_t* out_source_register_offset) {
  *out_source = LOOM_VALUE_ID_INVALID;
  *out_source_kind = LOOM_AMDGPU_FMA_MIX_SOURCE_F32;
  *out_source_register_offset = 0;

  const loom_op_t* defining_op =
      loom_amdgpu_source_defining_op(module, value_id);
  if (defining_op != NULL && loom_scalar_extf_isa(defining_op) &&
      loom_scalar_extf_result(defining_op) == value_id) {
    const loom_value_id_t input = loom_scalar_extf_input(defining_op);
    const loom_type_t input_type = loom_module_value_type(module, input);
    const loom_type_t result_type = loom_module_value_type(module, value_id);
    if (loom_amdgpu_type_is_f16(input_type) &&
        loom_amdgpu_type_is_f32(result_type)) {
      if (loom_amdgpu_select_fma_mix_packed_extract_source(
              module, input, out_source, out_source_kind,
              out_source_register_offset)) {
        return true;
      }
      *out_source = input;
      *out_source_kind = LOOM_AMDGPU_FMA_MIX_SOURCE_F16LO;
      return true;
    }
    return false;
  }

  const loom_type_t value_type = loom_module_value_type(module, value_id);
  if (loom_amdgpu_type_is_f32(value_type)) {
    *out_source = value_id;
    *out_source_kind = LOOM_AMDGPU_FMA_MIX_SOURCE_F32;
    *out_source_register_offset = 0;
    return true;
  }

  return false;
}

static bool loom_amdgpu_scalar_mulf_fastmath_allows_zero_add(
    const loom_op_t* source_op) {
  const uint8_t required_flags = LOOM_SCALAR_FASTMATHFLAGS_NNAN |
                                 LOOM_SCALAR_FASTMATHFLAGS_NSZ |
                                 LOOM_SCALAR_FASTMATHFLAGS_CONTRACT;
  return iree_all_bits_set(loom_scalar_mulf_fastmath(source_op),
                           required_flags);
}

static bool loom_amdgpu_vector_mulf_fastmath_allows_zero_add(
    const loom_op_t* source_op) {
  const uint8_t required_flags = LOOM_VECTOR_FASTMATHFLAGS_NNAN |
                                 LOOM_VECTOR_FASTMATHFLAGS_NSZ |
                                 LOOM_VECTOR_FASTMATHFLAGS_CONTRACT;
  return iree_all_bits_set(loom_vector_mulf_fastmath(source_op),
                           required_flags);
}

static bool loom_amdgpu_descriptor_ref_is_present(
    loom_low_lower_context_t* context, loom_amdgpu_descriptor_ref_t ref) {
  if (ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  return loom_amdgpu_descriptor_set_has_ref(descriptor_set, ref);
}

static loom_amdgpu_descriptor_ref_t loom_amdgpu_fma_mix_descriptor_ref_at(
    const loom_amdgpu_fma_mix_descriptor_ref_cube_t* refs,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds) {
  return (*refs)[source_kinds[0]][source_kinds[1]][source_kinds[2]];
}

static bool loom_amdgpu_select_fma_mix_descriptor_from_cubes(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds,
    const loom_amdgpu_fma_mix_descriptor_ref_cube_t* preferred_refs,
    const loom_amdgpu_fma_mix_descriptor_ref_cube_t* fallback_refs,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;

  for (uint32_t i = 0; i < 3; ++i) {
    IREE_ASSERT(loom_amdgpu_fma_mix_source_ref_is_valid(source_kinds[i]));
  }

  const loom_amdgpu_descriptor_ref_t preferred_ref =
      loom_amdgpu_fma_mix_descriptor_ref_at(preferred_refs, source_kinds);
  if (loom_amdgpu_descriptor_ref_is_present(context, preferred_ref)) {
    *out_descriptor_ref = preferred_ref;
    return true;
  }

  const loom_amdgpu_descriptor_ref_t fallback_ref =
      loom_amdgpu_fma_mix_descriptor_ref_at(fallback_refs, source_kinds);
  if (loom_amdgpu_descriptor_ref_is_present(context, fallback_ref)) {
    *out_descriptor_ref = fallback_ref;
    return true;
  }
  return false;
}

static bool loom_amdgpu_select_fma_mix_descriptor(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  return loom_amdgpu_select_fma_mix_descriptor_from_cubes(
      context, source_kinds, &kFmaMixF32DescriptorRefs,
      &kMadMixF32DescriptorRefs, out_descriptor_ref);
}

bool loom_amdgpu_select_fma_mix_half_result_descriptor(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds, bool high_result,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  return loom_amdgpu_select_fma_mix_descriptor_from_cubes(
      context, source_kinds,
      high_result ? &kFmaMixhiF16DescriptorRefs : &kFmaMixloF16DescriptorRefs,
      high_result ? &kMadMixhiF16DescriptorRefs : &kMadMixloF16DescriptorRefs,
      out_descriptor_ref);
}

static bool loom_amdgpu_select_fma_mix_zero_addend_descriptor(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref,
    bool* out_addend_literal_zero) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_addend_literal_zero = false;

  for (uint32_t i = 0; i < 3; ++i) {
    IREE_ASSERT(loom_amdgpu_fma_mix_source_ref_is_valid(source_kinds[i]));
  }

  if (source_kinds[2] == LOOM_AMDGPU_FMA_MIX_SOURCE_F32) {
    const loom_amdgpu_descriptor_ref_t literal_ref =
        kFmaMixF32Src2LiteralDescriptorRefs[source_kinds[0]][source_kinds[1]];
    if (loom_amdgpu_descriptor_ref_is_present(context, literal_ref)) {
      *out_descriptor_ref = literal_ref;
      *out_addend_literal_zero = true;
      return true;
    }
  }

  return loom_amdgpu_select_fma_mix_descriptor(context, source_kinds,
                                               out_descriptor_ref);
}

static void loom_amdgpu_reset_mulf_mix_plan(
    loom_amdgpu_mulf_mix_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_mulf_mix_plan_t){
      .sources = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID},
      .source_register_offsets = {0, 0},
      .result = LOOM_VALUE_ID_INVALID,
      .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      .addend_literal_zero = false,
      .source_kinds = {LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
                       LOOM_AMDGPU_FMA_MIX_SOURCE_F32},
      .lane_count = 0,
  };
}

static void loom_amdgpu_canonicalize_mulf_mix_sources(
    loom_value_id_t* sources, uint32_t* source_register_offsets,
    loom_amdgpu_fma_mix_source_kind_t* source_kinds) {
  if (source_kinds[0] != LOOM_AMDGPU_FMA_MIX_SOURCE_F32 ||
      !loom_amdgpu_fma_mix_source_is_f16(source_kinds[1])) {
    return;
  }
  const loom_value_id_t source = sources[0];
  sources[0] = sources[1];
  sources[1] = source;
  const uint32_t source_register_offset = source_register_offsets[0];
  source_register_offsets[0] = source_register_offsets[1];
  source_register_offsets[1] = source_register_offset;
  const loom_amdgpu_fma_mix_source_kind_t source_kind = source_kinds[0];
  source_kinds[0] = source_kinds[1];
  source_kinds[1] = source_kind;
}

static iree_string_view_t loom_amdgpu_descriptor_set_key(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL) {
    return IREE_SV("<missing>");
  }
  const iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset);
  return iree_string_view_is_empty(descriptor_set_key) ? IREE_SV("<empty>")
                                                       : descriptor_set_key;
}

static bool loom_amdgpu_source_value_has_exact_f32_immediate(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id) {
  if (fact_table == NULL) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, value_id);
  if (loom_type_is_vector(type)) {
    if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_F32) {
      return false;
    }
    loom_value_fact_uniform_element_t uniform = {0};
    if (!loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                                &uniform)) {
      return false;
    }
    facts = uniform.element;
  } else if (loom_type_is_scalar(type)) {
    if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_F32) {
      return false;
    }
  } else {
    return false;
  }
  return loom_value_facts_is_exact(facts) && loom_value_facts_is_float(facts);
}

typedef struct loom_amdgpu_fmaf_literal_operand_form_t {
  // Selected f32 FMA literal operand form.
  iree_string_view_t operand_form;
  // Source operand index carrying the exact f32 literal.
  uint32_t source_operand_index;
  // Diagnostic role for the literal operand.
  iree_string_view_t literal_role;
  // Descriptor key implementing the literal operand form.
  iree_string_view_t descriptor_key;
} loom_amdgpu_fmaf_literal_operand_form_t;

static const loom_amdgpu_fmaf_literal_operand_form_t
    kAmdgpuFmafLiteralOperandForms[] = {
        {
            .operand_form = IREE_SVL("fmaak"),
            .source_operand_index = 2,
            .literal_role = IREE_SVL("addend"),
            .descriptor_key = IREE_SVL("amdgpu.v_fmaak_f32"),
        },
        {
            .operand_form = IREE_SVL("fmamk"),
            .source_operand_index = 0,
            .literal_role = IREE_SVL("multiply_lhs"),
            .descriptor_key = IREE_SVL("amdgpu.v_fmamk_f32"),
        },
        {
            .operand_form = IREE_SVL("fmamk"),
            .source_operand_index = 1,
            .literal_role = IREE_SVL("multiply_rhs"),
            .descriptor_key = IREE_SVL("amdgpu.v_fmamk_f32"),
        },
};

static bool loom_amdgpu_fmaf_literal_operand_form(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fmaf_literal_operand_form_t* out_form) {
  *out_form = (loom_amdgpu_fmaf_literal_operand_form_t){0};

  loom_value_id_t sources[3] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  if (loom_scalar_fmaf_isa(source_op)) {
    sources[0] = loom_scalar_fmaf_a(source_op);
    sources[1] = loom_scalar_fmaf_b(source_op);
    sources[2] = loom_scalar_fmaf_c(source_op);
  } else if (loom_vector_fmaf_isa(source_op)) {
    sources[0] = loom_vector_fmaf_a(source_op);
    sources[1] = loom_vector_fmaf_b(source_op);
    sources[2] = loom_vector_fmaf_c(source_op);
  } else {
    return false;
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(kAmdgpuFmafLiteralOperandForms);
       ++i) {
    const loom_amdgpu_fmaf_literal_operand_form_t* form =
        &kAmdgpuFmafLiteralOperandForms[i];
    IREE_ASSERT_LT(form->source_operand_index, IREE_ARRAYSIZE(sources));
    if (loom_amdgpu_source_value_has_exact_f32_immediate(
            module, fact_table, sources[form->source_operand_index])) {
      *out_form = *form;
      return true;
    }
  }
  return false;
}

iree_status_t loom_amdgpu_emit_fmaf_literal_operand_form_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!iree_any_bit_set(loom_low_lower_context_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM)) {
    return iree_ok_status();
  }

  loom_amdgpu_fmaf_literal_operand_form_t literal_form = {0};
  if (!loom_amdgpu_fmaf_literal_operand_form(context, source_op,
                                             &literal_form)) {
    return iree_ok_status();
  }

  loom_low_lower_rule_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_select(
      context, &loom_amdgpu_arithmetic_lower_rule_set, source_op, &selection));
  if (selection.rule == NULL) {
    return iree_ok_status();
  }

  const loom_low_lower_descriptor_ref_t descriptor_ref =
      loom_low_lower_rule_first_descriptor_ref(
          &loom_amdgpu_arithmetic_lower_rule_set, selection.rule);
  if (descriptor_ref == LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(descriptor_ref,
                 loom_amdgpu_arithmetic_lower_rule_set.descriptor_ref_count);
  const iree_string_view_t descriptor_name =
      loom_amdgpu_arithmetic_lower_rule_set.descriptor_refs[descriptor_ref].key;
  const bool selected_literal =
      iree_string_view_equal(descriptor_name, IREE_SV("amdgpu.v_fmaak_f32")) ||
      iree_string_view_equal(descriptor_name, IREE_SV("amdgpu.v_fmamk_f32"));
  const bool selected_plain_fma =
      iree_string_view_equal(descriptor_name, IREE_SV("amdgpu.v_fma_f32"));
  if (!selected_literal && !selected_plain_fma) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const iree_string_view_t descriptor_set_name =
      loom_amdgpu_descriptor_set_key(descriptor_set);
  const iree_string_view_t decision =
      selected_literal ? IREE_SV("selected") : IREE_SV("rejected");
  const iree_string_view_t reason =
      selected_literal ? IREE_SV("literal_descriptor_selected")
                       : (loom_amdgpu_descriptor_set_has_key(
                              descriptor_set, literal_form.descriptor_key)
                              ? IREE_SV("literal_operand_contract_unmatched")
                              : IREE_SV("literal_descriptor_unavailable"));

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_string(descriptor_name),
      loom_param_string(literal_form.operand_form),
      loom_param_u32(literal_form.source_operand_index),
      loom_param_string(literal_form.literal_role),
      loom_param_string(descriptor_set_name),
      loom_param_string(IREE_SV("exact_f32_literal")),
      loom_param_string(decision),
      loom_param_string(reason),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_030_REF, params,
                                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_emit_mulf_mix_operand_form_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_mulf_mix_plan_t* plan) {
  if (!iree_any_bit_set(loom_low_lower_context_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM)) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, plan->descriptor_ref, &descriptor));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const iree_string_view_t descriptor_name = loom_low_descriptor_set_string(
      descriptor_set, descriptor.descriptor->key_string_offset);
  const iree_string_view_t descriptor_set_name =
      loom_amdgpu_descriptor_set_key(descriptor_set);
  const iree_string_view_t decision =
      plan->addend_literal_zero ? IREE_SV("selected") : IREE_SV("rejected");
  const iree_string_view_t reason =
      plan->addend_literal_zero ? IREE_SV("literal_descriptor_available")
                                : IREE_SV("literal_descriptor_unavailable");
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_string(descriptor_name),
      loom_param_u32(2),
      loom_param_string(descriptor_set_name),
      loom_param_string(IREE_SV("exact_f32_positive_zero")),
      loom_param_string(decision),
      loom_param_string(reason),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_027_REF, params,
                                       IREE_ARRAYSIZE(params));
}

static void loom_amdgpu_reset_packed_ternary_plan(
    loom_amdgpu_packed_ternary_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_packed_ternary_plan_t){
      .sources = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
                  LOOM_VALUE_ID_INVALID},
      .result = LOOM_VALUE_ID_INVALID,
      .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      .register_count = 0,
      .packet_unit_count = 0,
      .packet_count = 0,
  };
}

iree_status_t loom_amdgpu_select_vector_packed_fmaf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_packed_ternary_plan_t* out_plan, bool* out_selected) {
  loom_amdgpu_reset_packed_ternary_plan(out_plan);
  *out_selected = false;
  if (!loom_vector_fmaf_isa(source_op)) {
    return iree_ok_status();
  }

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t sources[3] = {
      loom_vector_fmaf_a(source_op),
      loom_vector_fmaf_b(source_op),
      loom_vector_fmaf_c(source_op),
  };
  const loom_value_id_t result = loom_vector_fmaf_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  for (uint32_t i = 0; i < 3; ++i) {
    const loom_type_t source_type = loom_module_value_type(module, sources[i]);
    if (!loom_type_equal(source_type, result_type)) {
      return iree_ok_status();
    }
  }

  uint32_t register_count = 0;
  uint32_t packet_unit_count = 0;
  uint32_t packet_count = 0;
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_value_id_t descriptor_sources[3] = {sources[0], sources[1], sources[2]};
  if (loom_amdgpu_type_is_even_packed_f16_vector(result_type,
                                                 &register_count)) {
    descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_FMAC_F16;
    bool descriptor_present =
        loom_amdgpu_descriptor_ref_is_present(context, descriptor_ref);
    if (!descriptor_present) {
      descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_FMA_F16;
      descriptor_present =
          loom_amdgpu_descriptor_ref_is_present(context, descriptor_ref);
    }
    if (!descriptor_present) {
      return iree_ok_status();
    }

    if (descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_FMAC_F16) {
      descriptor_sources[0] = sources[2];
      descriptor_sources[1] = sources[0];
      descriptor_sources[2] = sources[1];
    }
    packet_unit_count = 1;
    packet_count = register_count;
  } else if (loom_amdgpu_type_is_even_packed_f32_vector(result_type,
                                                        &register_count)) {
    descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_FMA_F32;
    if (!loom_amdgpu_descriptor_ref_is_present(context, descriptor_ref)) {
      return iree_ok_status();
    }
    packet_unit_count = 2;
    packet_count = register_count / packet_unit_count;
  } else {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_packed_ternary_plan_t){
      .sources = {descriptor_sources[0], descriptor_sources[1],
                  descriptor_sources[2]},
      .result = result,
      .descriptor_ref = descriptor_ref,
      .register_count = register_count,
      .packet_unit_count = packet_unit_count,
      .packet_count = packet_count,
  };
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_select_packed_fmai_descriptor(
    loom_low_lower_context_t* context, uint8_t overflow_flags,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;

  loom_amdgpu_descriptor_ref_t descriptor_refs[2] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MAD_I16,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MAD_U16,
  };
  if (iree_any_bit_set(overflow_flags, LOOM_VECTOR_INTOVERFLOWFLAGS_NUW)) {
    descriptor_refs[0] = LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MAD_U16;
    descriptor_refs[1] = LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MAD_I16;
  }

  for (uint32_t i = 0; i < IREE_ARRAYSIZE(descriptor_refs); ++i) {
    if (loom_amdgpu_descriptor_ref_is_present(context, descriptor_refs[i])) {
      *out_descriptor_ref = descriptor_refs[i];
      return true;
    }
  }
  return false;
}

iree_status_t loom_amdgpu_select_vector_packed_fmai_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_packed_ternary_plan_t* out_plan, bool* out_selected) {
  loom_amdgpu_reset_packed_ternary_plan(out_plan);
  *out_selected = false;
  if (!loom_vector_fmai_isa(source_op)) {
    return iree_ok_status();
  }

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t sources[3] = {
      loom_vector_fmai_a(source_op),
      loom_vector_fmai_b(source_op),
      loom_vector_fmai_c(source_op),
  };
  const loom_value_id_t result = loom_vector_fmai_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  uint32_t register_count = 0;
  if (!loom_amdgpu_type_is_even_packed_i16_vector(result_type,
                                                  &register_count)) {
    return iree_ok_status();
  }
  for (uint32_t i = 0; i < 3; ++i) {
    const loom_type_t source_type = loom_module_value_type(module, sources[i]);
    if (!loom_type_equal(source_type, result_type)) {
      return iree_ok_status();
    }
  }

  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_select_packed_fmai_descriptor(
          context, loom_vector_fmai_overflow(source_op), &descriptor_ref)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_packed_ternary_plan_t){
      .sources = {sources[0], sources[1], sources[2]},
      .result = result,
      .descriptor_ref = descriptor_ref,
      .register_count = register_count,
      .packet_unit_count = 1,
      .packet_count = register_count,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scalar_fmaf_mix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fma_mix_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_fma_mix_plan_t){
      .sources = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
                  LOOM_VALUE_ID_INVALID},
      .source_register_offsets = {0, 0, 0},
      .result = LOOM_VALUE_ID_INVALID,
      .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      .source_kinds = {LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
                       LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
                       LOOM_AMDGPU_FMA_MIX_SOURCE_F32},
  };
  *out_selected = false;
  if (!loom_scalar_fmaf_isa(source_op)) {
    return iree_ok_status();
  }

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t operands[3] = {
      loom_scalar_fmaf_a(source_op),
      loom_scalar_fmaf_b(source_op),
      loom_scalar_fmaf_c(source_op),
  };
  const loom_value_id_t result = loom_scalar_fmaf_result(source_op);
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }

  loom_value_id_t sources[3] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  uint32_t source_register_offsets[3] = {0, 0, 0};
  loom_amdgpu_fma_mix_source_kind_t source_kinds[3] = {
      LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
      LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
      LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
  };
  bool has_f16_source = false;
  for (uint32_t i = 0; i < 3; ++i) {
    if (!loom_amdgpu_select_fma_mix_source(module, operands[i], &sources[i],
                                           &source_kinds[i],
                                           &source_register_offsets[i])) {
      return iree_ok_status();
    }
    has_f16_source =
        has_f16_source || loom_amdgpu_fma_mix_source_is_f16(source_kinds[i]);
  }
  if (!has_f16_source) {
    return iree_ok_status();
  }

  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_select_fma_mix_descriptor(context, source_kinds,
                                             &descriptor_ref)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_fma_mix_plan_t){
      .sources = {sources[0], sources[1], sources[2]},
      .source_register_offsets = {source_register_offsets[0],
                                  source_register_offsets[1],
                                  source_register_offsets[2]},
      .result = result,
      .descriptor_ref = descriptor_ref,
      .source_kinds = {source_kinds[0], source_kinds[1], source_kinds[2]},
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scalar_mulf_mix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_mulf_mix_plan_t* out_plan, bool* out_selected) {
  loom_amdgpu_reset_mulf_mix_plan(out_plan);
  *out_selected = false;
  if (!loom_scalar_mulf_isa(source_op) ||
      !loom_amdgpu_scalar_mulf_fastmath_allows_zero_add(source_op)) {
    return iree_ok_status();
  }

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t operands[2] = {
      loom_scalar_mulf_lhs(source_op),
      loom_scalar_mulf_rhs(source_op),
  };
  const loom_value_id_t result = loom_scalar_mulf_result(source_op);
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }

  loom_value_id_t sources[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  uint32_t source_register_offsets[2] = {0, 0};
  loom_amdgpu_fma_mix_source_kind_t source_kinds[3] = {
      LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
      LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
      LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
  };
  bool has_f16_source = false;
  for (uint32_t i = 0; i < 2; ++i) {
    if (!loom_amdgpu_select_fma_mix_source(module, operands[i], &sources[i],
                                           &source_kinds[i],
                                           &source_register_offsets[i])) {
      return iree_ok_status();
    }
    has_f16_source =
        has_f16_source || loom_amdgpu_fma_mix_source_is_f16(source_kinds[i]);
  }
  if (!has_f16_source) {
    return iree_ok_status();
  }
  loom_amdgpu_canonicalize_mulf_mix_sources(sources, source_register_offsets,
                                            source_kinds);

  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  bool addend_literal_zero = false;
  if (!loom_amdgpu_select_fma_mix_zero_addend_descriptor(
          context, source_kinds, &descriptor_ref, &addend_literal_zero)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_mulf_mix_plan_t){
      .sources = {sources[0], sources[1]},
      .source_register_offsets = {source_register_offsets[0],
                                  source_register_offsets[1]},
      .result = result,
      .descriptor_ref = descriptor_ref,
      .addend_literal_zero = addend_literal_zero,
      .source_kinds = {source_kinds[0], source_kinds[1]},
      .lane_count = 1,
  };
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_select_vector_splatted_f16_mix_source(
    const loom_module_t* module, loom_value_id_t value_id,
    uint32_t expected_lane_count, loom_value_id_t* out_source,
    loom_amdgpu_fma_mix_source_kind_t* out_source_kind,
    uint32_t* out_source_register_offset) {
  *out_source = LOOM_VALUE_ID_INVALID;
  *out_source_kind = LOOM_AMDGPU_FMA_MIX_SOURCE_F32;
  *out_source_register_offset = 0;

  const loom_op_t* defining_op =
      loom_amdgpu_source_defining_op(module, value_id);
  if (defining_op == NULL || !loom_vector_splat_isa(defining_op) ||
      loom_vector_splat_result(defining_op) != value_id) {
    return false;
  }
  const loom_type_t value_type = loom_module_value_type(module, value_id);
  if (loom_amdgpu_vector_f32_lane_count(value_type) != expected_lane_count) {
    return false;
  }

  if (!loom_amdgpu_select_fma_mix_source(
          module, loom_vector_splat_scalar(defining_op), out_source,
          out_source_kind, out_source_register_offset)) {
    return false;
  }
  return loom_amdgpu_fma_mix_source_is_f16(*out_source_kind);
}

iree_status_t loom_amdgpu_select_vector_mulf_mix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_mulf_mix_plan_t* out_plan, bool* out_selected) {
  loom_amdgpu_reset_mulf_mix_plan(out_plan);
  *out_selected = false;
  if (!loom_vector_mulf_isa(source_op) ||
      !loom_amdgpu_vector_mulf_fastmath_allows_zero_add(source_op)) {
    return iree_ok_status();
  }

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t lhs = loom_vector_mulf_lhs(source_op);
  const loom_value_id_t rhs = loom_vector_mulf_rhs(source_op);
  const loom_value_id_t result = loom_vector_mulf_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const uint32_t lane_count = loom_amdgpu_vector_f32_lane_count(result_type);
  if (lane_count == 0) {
    return iree_ok_status();
  }

  loom_value_id_t vector_source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t splat_source = LOOM_VALUE_ID_INVALID;
  uint32_t splat_source_register_offset = 0;
  loom_amdgpu_fma_mix_source_kind_t splat_source_kind =
      LOOM_AMDGPU_FMA_MIX_SOURCE_F32;
  const bool lhs_is_vector =
      loom_amdgpu_vector_f32_lane_count(loom_module_value_type(module, lhs)) ==
      lane_count;
  const bool rhs_is_vector =
      loom_amdgpu_vector_f32_lane_count(loom_module_value_type(module, rhs)) ==
      lane_count;
  if (lhs_is_vector && loom_amdgpu_select_vector_splatted_f16_mix_source(
                           module, rhs, lane_count, &splat_source,
                           &splat_source_kind, &splat_source_register_offset)) {
    vector_source = lhs;
  } else if (rhs_is_vector &&
             loom_amdgpu_select_vector_splatted_f16_mix_source(
                 module, lhs, lane_count, &splat_source, &splat_source_kind,
                 &splat_source_register_offset)) {
    vector_source = rhs;
  } else {
    return iree_ok_status();
  }

  loom_value_id_t sources[2] = {
      splat_source,
      vector_source,
  };
  uint32_t source_register_offsets[2] = {splat_source_register_offset, 0};
  loom_amdgpu_fma_mix_source_kind_t source_kinds[3] = {
      splat_source_kind,
      LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
      LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
  };
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  bool addend_literal_zero = false;
  if (!loom_amdgpu_select_fma_mix_zero_addend_descriptor(
          context, source_kinds, &descriptor_ref, &addend_literal_zero)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_mulf_mix_plan_t){
      .sources = {sources[0], sources[1]},
      .source_register_offsets = {source_register_offsets[0],
                                  source_register_offsets[1]},
      .result = result,
      .descriptor_ref = descriptor_ref,
      .addend_literal_zero = addend_literal_zero,
      .source_kinds = {source_kinds[0], source_kinds[1]},
      .lane_count = lane_count,
  };
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_extract_fma_mix_register_unit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t register_offset,
    loom_value_id_t* out_source) {
  *out_source = low_source;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_source_type =
      loom_module_value_type(module, low_source);
  const uint32_t unit_count =
      loom_low_register_type_unit_count(low_source_type);
  if (unit_count == 1 && register_offset == 0) {
    return iree_ok_status();
  }
  if (unit_count == 0 || register_offset >= unit_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU mixed-FMA source selected an invalid "
                            "packed register unit");
  }
  const loom_type_t unit_type =
      loom_low_register_type_with_unit_count(low_source_type, 1);
  return loom_amdgpu_emit_low_slice(context, source_op, low_source,
                                    register_offset, unit_type, out_source);
}

static iree_status_t loom_amdgpu_lookup_fma_mix_packet_sources(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan, loom_value_id_t out_operands[3]) {
  for (uint32_t i = 0; i < 3; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->sources[i],
                                                     &out_operands[i]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_fma_mix_register_unit(
        context, source_op, out_operands[i], plan->source_register_offsets[i],
        &out_operands[i]));
    if (loom_amdgpu_fma_mix_source_is_f16(plan->source_kinds[i])) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, out_operands[i], &out_operands[i]));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_fma_mix_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan, loom_type_t result_type,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[3] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_fma_mix_packet_sources(
      context, source_op, plan, operands));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, plan->descriptor_ref, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
      &result_type, 1, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_tied_fma_mix_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan, loom_value_id_t accumulator,
    loom_type_t result_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t sources[3] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_fma_mix_packet_sources(
      context, source_op, plan, sources));
  const loom_value_id_t operands[] = {
      accumulator,
      sources[0],
      sources[1],
      sources[2],
  };
  const loom_tied_result_t tied_results[] = {
      {
          .result_index = 0,
          .operand_index = 0,
          .has_type_change = false,
      },
  };

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, plan->descriptor_ref, &descriptor));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, 1, tied_results,
      IREE_ARRAYSIZE(tied_results), source_op->location, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_scalar_fmaf_mix(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_packet(context, source_op, plan,
                                                       result_type, &result));
  return loom_low_lower_bind_value(context, plan->result, result);
}

static iree_status_t loom_amdgpu_packed_ternary_packet_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t register_offset,
    loom_type_t packet_type, loom_value_id_t* out_source) {
  *out_source = low_source;
  const uint32_t unit_count =
      loom_low_register_type_unit_count(loom_module_value_type(
          loom_low_lower_context_module(context), low_source));
  const uint32_t packet_unit_count =
      loom_low_register_type_unit_count(packet_type);
  if (unit_count == 0 || packet_unit_count == 0 ||
      register_offset > unit_count ||
      packet_unit_count > unit_count - register_offset) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU packed ternary selected an invalid packed "
                            "register range");
  }
  if (unit_count == packet_unit_count && register_offset == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source,
                                    register_offset, packet_type, out_source);
}

static iree_status_t loom_amdgpu_emit_packed_ternary_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_type_t packet_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_tied_result_t tied_accumulator[] = {
      {
          .result_index = 0,
          .operand_index = 0,
          .has_type_change = false,
      },
  };
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  if (descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_FMAC_F16) {
    tied_results = tied_accumulator;
    tied_result_count = IREE_ARRAYSIZE(tied_accumulator);
  }

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_named_attr_slice_empty(), &packet_type, 1, tied_results,
      tied_result_count, source_op->location, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_packed_ternary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_packed_ternary_plan_t* plan) {
  if (plan->packet_count == 0 || plan->packet_unit_count == 0 ||
      plan->packet_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS ||
      plan->packet_count > UINT32_MAX / plan->packet_unit_count ||
      plan->register_count != plan->packet_count * plan->packet_unit_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU packed ternary selected an invalid packet shape");
  }

  loom_value_id_t low_sources[3] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < 3; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->sources[i],
                                                     &low_sources[i]));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  const loom_type_t packet_type = loom_low_register_type_with_unit_count(
      result_type, plan->packet_unit_count);

  loom_value_id_t packet_results[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t packet_index = 0; packet_index < plan->packet_count;
       ++packet_index) {
    loom_value_id_t operands[3] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
    };
    const uint32_t register_offset = packet_index * plan->packet_unit_count;
    for (uint32_t i = 0; i < 3; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_packed_ternary_packet_source(
          context, source_op, low_sources[i], register_offset, packet_type,
          &operands[i]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_ternary_packet(
        context, source_op, plan->descriptor_ref, operands,
        IREE_ARRAYSIZE(operands), packet_type, &packet_results[packet_index]));
  }

  if (plan->packet_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, packet_results[0]);
  }
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), packet_results,
      plan->packet_count, result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static bool loom_amdgpu_mulf_mix_source_is_vector(
    loom_low_lower_context_t* context, const loom_amdgpu_mulf_mix_plan_t* plan,
    uint32_t source_index) {
  if (plan->lane_count == 1) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(
      loom_low_lower_context_module(context), plan->sources[source_index]);
  return loom_amdgpu_vector_f32_lane_count(type) == plan->lane_count;
}

static iree_status_t loom_amdgpu_mulf_mix_lane_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_mulf_mix_plan_t* plan, const loom_value_id_t* low_sources,
    uint32_t source_index, uint32_t lane, loom_type_t lane_type,
    loom_value_id_t* out_source) {
  *out_source = low_sources[source_index];
  if (loom_amdgpu_mulf_mix_source_is_vector(context, plan, source_index)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_sources[source_index], lane, lane_type,
        out_source));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_fma_mix_register_unit(
      context, source_op, *out_source,
      plan->source_register_offsets[source_index], out_source));
  if (loom_amdgpu_fma_mix_source_is_f16(plan->source_kinds[source_index])) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, *out_source, out_source));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_mulf_mix(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_mulf_mix_plan_t* plan) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_mulf_mix_operand_form_diagnostic(
      context, source_op, plan));

  loom_value_id_t low_sources[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < 2; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->sources[i],
                                                     &low_sources[i]));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_type_t lane_type = result_type;
  if (plan->lane_count > 1) {
    lane_type = loom_low_register_type_with_unit_count(result_type, 1);
  }

  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  if (plan->addend_literal_zero) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_intern(context, IREE_SV("imm32"), &attrs[0].name_id));
    attrs[0].value = loom_attr_i64(0);
    attr_count = 1;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
        &zero));
  }

  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t lane = 0; lane < plan->lane_count; ++lane) {
    loom_value_id_t operands[3] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
        zero,
    };
    for (uint32_t i = 0; i < 2; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_mulf_mix_lane_source(
          context, source_op, plan, low_sources, i, lane, lane_type,
          &operands[i]));
    }
    const iree_host_size_t operand_count = plan->addend_literal_zero ? 2 : 3;
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, plan->descriptor_ref, operands, operand_count,
        loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1, &low_op));
    lane_results[lane] = loom_value_slice_get(loom_low_op_results(low_op), 0);
  }

  if (plan->lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, lane_results[0]);
  }
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, plan->lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

void loom_amdgpu_mark_fma_mix_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan) {
  (void)source_op;
  for (uint32_t i = 0; i < 3; ++i) {
    loom_low_lower_require_source_value_storage(context, plan->sources[i]);
  }
}

void loom_amdgpu_mark_packed_ternary_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_packed_ternary_plan_t* plan) {
  (void)source_op;
  for (uint32_t i = 0; i < 3; ++i) {
    loom_low_lower_require_source_value_storage(context, plan->sources[i]);
  }
}

void loom_amdgpu_mark_mulf_mix_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_mulf_mix_plan_t* plan) {
  (void)source_op;
  for (uint32_t i = 0; i < 2; ++i) {
    loom_low_lower_require_source_value_storage(context, plan->sources[i]);
  }
}
