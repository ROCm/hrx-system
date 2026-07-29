// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/value/vector_conversion.h"

#include <stdint.h>

#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/scalarization.h"
#include "loom/target/arch/amdgpu/lower/arithmetic.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/value/scalar_conversion.h"

static void loom_amdgpu_vector_extract_plan_from_accepted_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_extract_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_extract_plan_t){0};
  IREE_ASSERT(loom_vector_extract_isa(source_op));
  loom_attribute_t static_indices =
      loom_vector_extract_static_indices(source_op);
  IREE_ASSERT_EQ(static_indices.kind, LOOM_ATTR_I64_ARRAY);
  const loom_value_slice_t indices = loom_vector_extract_indices(source_op);

  const loom_value_id_t source = loom_vector_extract_source(source_op);
  const loom_value_id_t result = loom_vector_extract_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);

  loom_amdgpu_vector_storage_t source_storage = {0};
  const bool source_storage_matches =
      loom_amdgpu_type_vector_storage(source_type, &source_storage);
  IREE_ASSERT(source_storage_matches);

  uint32_t result_register_count = 0;
  uint32_t result_lane_count = 1;
  bool sign_extend_packed_lane = false;
  if (loom_type_is_scalar(result_type)) {
    IREE_ASSERT_EQ(loom_type_element_type(result_type),
                   source_storage.element_type);
    switch (source_storage.kind) {
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_8BIT_FLOAT:
        result_register_count = source_storage.element_register_count;
        break;
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER:
        result_register_count = source_storage.element_register_count;
        sign_extend_packed_lane = true;
        break;
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_NONE:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK:
      default:
        IREE_ASSERT_UNREACHABLE(
            "accepted AMDGPU vector.extract has unsupported source storage");
        IREE_BUILTIN_UNREACHABLE();
    }
  } else {
    loom_amdgpu_vector_storage_t result_storage = {0};
    const bool result_storage_matches =
        loom_amdgpu_type_vector_storage(result_type, &result_storage);
    IREE_ASSERT(result_storage_matches);
    IREE_ASSERT_EQ(result_storage.kind, source_storage.kind);
    IREE_ASSERT_EQ(result_storage.element_type, source_storage.element_type);
    switch (source_storage.kind) {
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT:
        result_register_count = result_storage.register_count;
        result_lane_count = result_storage.element_count;
        break;
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_NONE:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_8BIT_FLOAT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER:
      default:
        IREE_ASSERT_UNREACHABLE(
            "accepted AMDGPU vector.extract has unsupported result storage");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  IREE_ASSERT_NE(source_storage.element_count, 0u);
  IREE_ASSERT_NE(source_storage.register_count, 0u);
  IREE_ASSERT_NE(result_register_count, 0u);

  IREE_ASSERT_LE(static_indices.count, loom_type_rank(source_type));
  if (loom_type_is_scalar(result_type)) {
    IREE_ASSERT_EQ(static_indices.count, loom_type_rank(source_type));
  } else {
    IREE_ASSERT_EQ(static_indices.count + loom_type_rank(result_type),
                   loom_type_rank(source_type));
  }

  bool is_dynamic = false;
  uint32_t lane_offset = 0;
  loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
  if (static_indices.count == 1 && static_indices.i64_array[0] == INT64_MIN) {
    IREE_ASSERT_EQ(indices.count, 1u);
    IREE_ASSERT(loom_type_is_scalar(result_type));
    is_dynamic = true;
    dynamic_index = indices.values[0];
  } else {
    IREE_ASSERT_EQ(indices.count, 0u);
    int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
    for (uint16_t i = 0; i < static_indices.count; ++i) {
      const int64_t index = static_indices.i64_array[i];
      IREE_ASSERT_GE(index, 0);
      source_indices[i] = index;
    }
    const bool has_static_lane_offset =
        loom_amdgpu_static_vector_flat_register_from_indices(
            source_type, source_indices, &lane_offset);
    IREE_ASSERT(has_static_lane_offset);
    IREE_ASSERT_LE((uint64_t)lane_offset + result_lane_count,
                   (uint64_t)source_storage.element_count);
  }

  *out_plan = (loom_amdgpu_vector_extract_plan_t){
      .source = source,
      .dynamic_index = dynamic_index,
      .result = result,
      .lane_offset = lane_offset,
      .lane_count = source_storage.element_count,
      .register_count = source_storage.register_count,
      .result_register_count = result_register_count,
      .element_register_count = source_storage.element_register_count,
      .lane_bit_count = source_storage.element_bit_count,
      .sign_extend_packed_lane = sign_extend_packed_lane,
      .is_dynamic = is_dynamic,
  };
}

iree_status_t loom_amdgpu_select_vector_extract_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_extract_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_extract_plan_t){0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_select_arithmetic_contract(context, source_op, out_selected));
  if (*out_selected) {
    loom_amdgpu_vector_extract_plan_from_accepted_op(
        loom_low_lower_context_module(context), source_op, out_plan);
  }
  return iree_ok_status();
}

enum {
  LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SIGN_EXTEND_PACKED_SOURCE = 1u << 0,
  LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SOURCE_WIDER_THAN_RESULT = 1u << 1,
};
typedef uint8_t loom_amdgpu_vector_conversion_lane_rule_flags_t;

typedef struct loom_amdgpu_vector_conversion_lane_rule_t {
  // Descriptor emitted by lanes that perform a numeric conversion packet.
  loom_amdgpu_descriptor_ref_t convert_descriptor_ref;
  // Allowed source scalar element types.
  loom_scalar_type_set_t source_element_types;
  // Allowed result scalar element types.
  loom_scalar_type_set_t result_element_types;
  // Bitfield of LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_* flags.
  loom_amdgpu_vector_conversion_lane_rule_flags_t flags;
} loom_amdgpu_vector_conversion_lane_rule_t;

#define LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(                          \
    scalar_op, descriptor_ref_, source_element_types_, result_element_types_, \
    flags_)                                                                   \
  [LOOM_AMDGPU_SCALAR_CONVERSION_OP_##scalar_op] = {                          \
      .convert_descriptor_ref = (descriptor_ref_),                            \
      .source_element_types = (source_element_types_),                        \
      .result_element_types = (result_element_types_),                        \
      .flags = (flags_),                                                      \
  }

static const loom_amdgpu_vector_conversion_lane_rule_t
    kAmdgpuVectorConversionLaneRulesByScalarOp
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_] = {
            LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(
                TRUNCI, LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD,
                LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD,
                LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SOURCE_WIDER_THAN_RESULT),
            LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(
                SITOFP, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_I32,
                LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD_LE32,
                LOOM_SCALAR_TYPE_SET_F32,
                LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SIGN_EXTEND_PACKED_SOURCE),
            LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(
                UITOFP, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32,
                LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD_LE32,
                LOOM_SCALAR_TYPE_SET_F32, 0),
            LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(
                FPTOSI, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                LOOM_SCALAR_TYPE_SET_F32,
                LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD_LE32, 0),
            LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(
                FPTOUI, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                LOOM_SCALAR_TYPE_SET_F32,
                LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD_LE32, 0),
};

#undef LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW

static const loom_amdgpu_vector_conversion_lane_rule_t*
loom_amdgpu_vector_conversion_lane_rule(
    loom_op_kind_t op_kind, loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type) {
  const loom_vector_scalarization_t* scalarization =
      loom_vector_scalarization_lookup(op_kind);
  if (scalarization == NULL) {
    return NULL;
  }
  const loom_amdgpu_scalar_conversion_op_group_t op_group =
      loom_amdgpu_scalar_conversion_op_group(scalarization->lane_op_kind);
  if (op_group >= IREE_ARRAYSIZE(kAmdgpuVectorConversionLaneRulesByScalarOp)) {
    return NULL;
  }
  const loom_amdgpu_vector_conversion_lane_rule_t* rule =
      &kAmdgpuVectorConversionLaneRulesByScalarOp[op_group];
  if (!loom_scalar_type_set_contains(rule->source_element_types,
                                     source_element_type) ||
      !loom_scalar_type_set_contains(rule->result_element_types,
                                     result_element_type)) {
    return NULL;
  }
  if (iree_all_bits_set(
          rule->flags,
          LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SOURCE_WIDER_THAN_RESULT) &&
      loom_amdgpu_integer_scalar_type_bit_count(source_element_type) <=
          loom_amdgpu_integer_scalar_type_bit_count(result_element_type)) {
    return NULL;
  }
  return rule;
}

typedef uint8_t loom_amdgpu_vector_conversion_kind_flags_t;

enum {
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_SOURCE = 1u << 0,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT = 1u << 1,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION = 1u << 2,
};

static bool loom_amdgpu_vector_conversion_can_use_packed_i8_permute(
    const loom_amdgpu_vector_storage_t* result_storage) {
  return result_storage->kind ==
             LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER &&
         result_storage->element_bit_count == 8 &&
         result_storage->element_count == result_storage->register_count * 4u;
}

static const loom_amdgpu_vector_conversion_kind_t kAmdgpuVectorConversionKindByStorage
    [LOOM_AMDGPU_VECTOR_STORAGE_KIND_COUNT_]
    [LOOM_AMDGPU_VECTOR_STORAGE_KIND_COUNT_] = {
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_FULL_32,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_PACKED_INTEGER,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_FULL_32,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_PACKED_INTEGER,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_FULL_32,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER,
};

static const loom_amdgpu_vector_conversion_kind_flags_t
    kAmdgpuVectorConversionKindFlags[LOOM_AMDGPU_VECTOR_CONVERSION_KIND_COUNT_] = {
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_FULL_32] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION,
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_FULL_32] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION,
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_PACKED_INTEGER] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT |
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION,
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_PACKED_INTEGER] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT |
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION,
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_FULL_32] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_SOURCE,
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_SOURCE |
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT,
};

static const loom_amdgpu_descriptor_ref_t kAmdgpuPackedU8ToF32LaneDescriptorRefs
    [LOOM_AMDGPU_PACKED_I8_LANES_PER_REGISTER] = {
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_UBYTE0,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_UBYTE1,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_UBYTE2,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_UBYTE3,
};

static loom_amdgpu_vector_conversion_kind_flags_t
loom_amdgpu_vector_conversion_kind_flags(
    loom_amdgpu_vector_conversion_kind_t kind) {
  if (kind >= LOOM_AMDGPU_VECTOR_CONVERSION_KIND_COUNT_) {
    return 0;
  }
  return kAmdgpuVectorConversionKindFlags[kind];
}

static bool loom_amdgpu_vector_conversion_select_storage_kind(
    const loom_amdgpu_vector_storage_t* source_storage,
    const loom_amdgpu_vector_storage_t* result_storage,
    loom_amdgpu_vector_conversion_kind_t* out_kind) {
  *out_kind = LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE;
  if (source_storage->kind >= LOOM_AMDGPU_VECTOR_STORAGE_KIND_COUNT_ ||
      result_storage->kind >= LOOM_AMDGPU_VECTOR_STORAGE_KIND_COUNT_) {
    return false;
  }
  *out_kind = kAmdgpuVectorConversionKindByStorage[source_storage->kind]
                                                  [result_storage->kind];
  return *out_kind != LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE;
}

static bool loom_amdgpu_vector_conversion_descriptor_refs_present(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_vector_conversion_kind_t kind,
    loom_amdgpu_descriptor_ref_t convert_descriptor_ref,
    bool sign_extend_packed_source) {
  const loom_amdgpu_vector_conversion_kind_flags_t kind_flags =
      loom_amdgpu_vector_conversion_kind_flags(kind);
  if (convert_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      !loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          convert_descriptor_ref)) {
    return false;
  }

  static const loom_amdgpu_descriptor_ref_t kFullSourceRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
  };
  if (convert_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      iree_any_bit_set(
          kind_flags,
          LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION) &&
      !loom_amdgpu_descriptor_set_has_all_refs(
          descriptor_set, kFullSourceRefs, IREE_ARRAYSIZE(kFullSourceRefs))) {
    return false;
  }

  static const loom_amdgpu_descriptor_ref_t kPackedResultRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
  };
  if (iree_any_bit_set(kind_flags,
                       LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT) &&
      !loom_amdgpu_descriptor_set_has_all_refs(
          descriptor_set, kPackedResultRefs,
          IREE_ARRAYSIZE(kPackedResultRefs))) {
    return false;
  }

  static const loom_amdgpu_descriptor_ref_t kPackedSourceSignedRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
  };
  static const loom_amdgpu_descriptor_ref_t kPackedSourceUnsignedRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
  };
  if (!iree_any_bit_set(kind_flags,
                        LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_SOURCE)) {
    return true;
  }
  if (sign_extend_packed_source) {
    return loom_amdgpu_descriptor_set_has_all_refs(
        descriptor_set, kPackedSourceSignedRefs,
        IREE_ARRAYSIZE(kPackedSourceSignedRefs));
  }
  return loom_amdgpu_descriptor_set_has_all_refs(
      descriptor_set, kPackedSourceUnsignedRefs,
      IREE_ARRAYSIZE(kPackedSourceUnsignedRefs));
}

static bool loom_amdgpu_select_vector_conversion_plan_for_op(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_vector_conversion_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_conversion_plan_t){0};
  if (source_op->operand_count != 1 || source_op->result_count != 1) {
    return false;
  }

  const loom_value_id_t source = loom_op_const_operands(source_op)[0];
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  loom_amdgpu_vector_storage_t source_storage = {0};
  loom_amdgpu_vector_storage_t result_storage = {0};
  if (!loom_amdgpu_type_vector_storage(loom_module_value_type(module, source),
                                       &source_storage) ||
      !loom_amdgpu_type_vector_storage(loom_module_value_type(module, result),
                                       &result_storage) ||
      source_storage.element_count != result_storage.element_count) {
    return false;
  }

  const loom_amdgpu_vector_conversion_lane_rule_t* lane_rule =
      loom_amdgpu_vector_conversion_lane_rule(source_op->kind,
                                              source_storage.element_type,
                                              result_storage.element_type);
  if (lane_rule == NULL) {
    return false;
  }
  const bool sign_extend_packed_source = iree_all_bits_set(
      lane_rule->flags,
      LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SIGN_EXTEND_PACKED_SOURCE);

  loom_amdgpu_vector_conversion_kind_t kind =
      LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE;
  if (!loom_amdgpu_vector_conversion_select_storage_kind(
          &source_storage, &result_storage, &kind)) {
    return false;
  }
  if (kind == LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_FULL_32 &&
      source_storage.element_type == LOOM_SCALAR_TYPE_I8 &&
      result_storage.element_type == LOOM_SCALAR_TYPE_F32 &&
      lane_rule->convert_descriptor_ref ==
          LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32 &&
      loom_amdgpu_descriptor_set_has_all_refs(
          descriptor_set, kAmdgpuPackedU8ToF32LaneDescriptorRefs,
          IREE_ARRAYSIZE(kAmdgpuPackedU8ToF32LaneDescriptorRefs))) {
    kind = LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_U8_TO_F32;
  }
  if (kind != LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_U8_TO_F32 &&
      !loom_amdgpu_vector_conversion_descriptor_refs_present(
          descriptor_set, kind, lane_rule->convert_descriptor_ref,
          sign_extend_packed_source)) {
    return false;
  }

  loom_amdgpu_i8_pack_permute_plan_t packed_i8_permute = {0};
  if (loom_amdgpu_vector_conversion_can_use_packed_i8_permute(
          &result_storage)) {
    loom_amdgpu_select_i8_pack_permute_plan(descriptor_set, &packed_i8_permute);
  }

  *out_plan = (loom_amdgpu_vector_conversion_plan_t){
      .source = source,
      .result = result,
      .kind = kind,
      .source_element_type = source_storage.element_type,
      .result_element_type = result_storage.element_type,
      .source_bit_count = source_storage.element_bit_count,
      .result_bit_count = result_storage.element_bit_count,
      .lane_count = source_storage.element_count,
      .source_register_count = source_storage.register_count,
      .result_register_count = result_storage.register_count,
      .source_element_register_count = source_storage.element_register_count,
      .convert_descriptor_ref =
          kind == LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_U8_TO_F32
              ? LOOM_AMDGPU_DESCRIPTOR_REF_NONE
              : lane_rule->convert_descriptor_ref,
      .packed_i8_permute = packed_i8_permute,
      .sign_extend_packed_source = sign_extend_packed_source,
  };
  return true;
}

iree_status_t loom_amdgpu_select_vector_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_conversion_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_select_vector_conversion_plan_for_op(
      loom_low_lower_context_module(context),
      loom_low_lower_context_descriptor_set(context), source_op, out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_vector_conversion(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }

  loom_amdgpu_vector_conversion_plan_t plan = {0};
  if (!loom_amdgpu_select_vector_conversion_plan_for_op(
          loom_target_low_legality_module(context),
          loom_target_low_legality_descriptor_set(context), op, &plan)) {
    return iree_ok_status();
  }
  *out_handled = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_extract_packed_register_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, const loom_amdgpu_vector_extract_plan_t* plan,
    uint32_t lane_offset, loom_amdgpu_bitfield_extract_mode_t mode,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(plan->lane_bit_count == 8 || plan->lane_bit_count == 16);
  const uint32_t lanes_per_register = 32u / plan->lane_bit_count;
  const uint32_t register_offset = lane_offset / lanes_per_register;
  const uint32_t register_bit_offset =
      (lane_offset % lanes_per_register) * plan->lane_bit_count;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, plan->register_count, register_offset,
      lane_type, &source_register));
  return loom_amdgpu_extract_vgpr_bitfield(
      context, source_op, source_register, register_bit_offset,
      plan->lane_bit_count, mode, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_extract_vector_conversion_packed_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, uint32_t lane_index, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_vector_extract_plan_t extract_plan = {
      .source = plan->source,
      .result = plan->result,
      .lane_count = plan->lane_count,
      .register_count = plan->source_register_count,
      .result_register_count = 1,
      .element_register_count = 1,
      .lane_bit_count = plan->source_bit_count,
      .sign_extend_packed_lane = plan->sign_extend_packed_source,
  };
  loom_amdgpu_bitfield_extract_mode_t extract_mode =
      plan->sign_extend_packed_source
          ? LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND
          : LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_RAW_SHIFTED;
  if (plan->kind ==
          LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER &&
      plan->convert_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return loom_amdgpu_extract_packed_register_lane(
        context, source_op, low_source, &extract_plan, lane_index, extract_mode,
        lane_type, out_lane);
  }
  if (!plan->sign_extend_packed_source) {
    extract_mode = LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_ZERO_EXTEND;
  }
  return loom_amdgpu_extract_packed_register_lane(
      context, source_op, low_source, &extract_plan, lane_index, extract_mode,
      lane_type, out_lane);
}

static iree_status_t loom_amdgpu_extract_vector_conversion_full_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, uint32_t lane_index,
    loom_type_t source_lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t register_offset =
      lane_index * plan->source_element_register_count;
  return loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, plan->source_register_count,
      register_offset, source_lane_type, out_lane);
}

static iree_status_t loom_amdgpu_extract_vector_conversion_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, uint32_t lane_index,
    loom_type_t source_lane_type, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const loom_amdgpu_vector_conversion_kind_flags_t kind_flags =
      loom_amdgpu_vector_conversion_kind_flags(plan->kind);
  if (iree_any_bit_set(kind_flags,
                       LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_SOURCE)) {
    return loom_amdgpu_extract_vector_conversion_packed_lane(
        context, source_op, plan, low_source, lane_index, lane_type, out_lane);
  }
  if (plan->kind != LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE &&
      plan->kind < LOOM_AMDGPU_VECTOR_CONVERSION_KIND_COUNT_) {
    return loom_amdgpu_extract_vector_conversion_full_lane(
        context, source_op, plan, low_source, lane_index, source_lane_type,
        out_lane);
  }
  IREE_ASSERT_UNREACHABLE("invalid AMDGPU vector conversion source kind");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_convert_vector_conversion_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan, loom_value_id_t lane,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (plan->convert_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    *out_lane = lane;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_materialize_low_vgpr_b32(context, source_op, lane, &lane));
  return loom_amdgpu_emit_vgpr_unary(context, source_op,
                                     plan->convert_descriptor_ref, lane,
                                     lane_type, out_lane);
}

static iree_status_t loom_amdgpu_lower_vector_conversion_full_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type) {
  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_conversion_lane(
        context, source_op, plan, low_source, i, source_lane_type, lane_type,
        &source_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_convert_vector_conversion_lane(
        context, source_op, plan, source_lane, lane_type, &lanes[i]));
  }
  return loom_amdgpu_bind_low_register_range(
      context, source_op, plan->result, lanes, plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_vector_conversion_packed_u8_to_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type) {
  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint32_t lane_index = 0;
  for (uint32_t register_index = 0;
       register_index < plan->source_register_count; ++register_index) {
    loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->source_register_count,
        register_index, source_lane_type, &source_register));
    for (uint32_t register_lane = 0;
         register_lane < LOOM_AMDGPU_PACKED_I8_LANES_PER_REGISTER &&
         lane_index < plan->lane_count;
         ++register_lane, ++lane_index) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op,
          kAmdgpuPackedU8ToF32LaneDescriptorRefs[register_lane],
          source_register, lane_type, &lanes[lane_index]));
    }
  }
  return loom_amdgpu_bind_low_register_range(
      context, source_op, plan->result, lanes, plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_vector_conversion_packed_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type) {
  if (plan->packed_i8_permute.kind != LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_NONE) {
    loom_value_id_t converted_lanes[LOOM_AMDGPU_MAX_PACKED_I8_LANES];
    for (uint32_t i = 0; i < plan->lane_count; ++i) {
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_conversion_lane(
          context, source_op, plan, low_source, i, source_lane_type, lane_type,
          &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_convert_vector_conversion_lane(
          context, source_op, plan, source_lane, lane_type,
          &converted_lanes[i]));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, converted_lanes[i], &converted_lanes[i]));
    }

    loom_value_id_t result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
    IREE_RETURN_IF_ERROR(loom_amdgpu_pack_i8_lanes_with_permute(
        context, source_op, &plan->packed_i8_permute, converted_lanes,
        plan->lane_count, lane_type, result_registers));
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               result_registers,
                                               plan->result_register_count);
  }

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  const uint32_t lanes_per_register = 32u / plan->result_bit_count;
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * lanes_per_register;
    for (uint32_t register_lane = 0; register_lane < lanes_per_register;
         ++register_lane) {
      const uint32_t lane_index = lane_base + register_lane;
      if (lane_index >= plan->lane_count) {
        break;
      }
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_conversion_lane(
          context, source_op, plan, low_source, lane_index, source_lane_type,
          lane_type, &source_lane));
      loom_value_id_t converted_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_convert_vector_conversion_lane(
          context, source_op, plan, source_lane, lane_type, &converted_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, converted_lane, &converted_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_pack_lane_bits_into_register(
          context, source_op, converted_lane, plan->result_bit_count,
          register_lane * plan->result_bit_count, lane_type, &packed));
    }
    result_registers[register_index] = packed;
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_lower_vector_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  if (plan->kind == LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_U8_TO_F32) {
    return loom_amdgpu_lower_vector_conversion_packed_u8_to_f32(
        context, source_op, plan, low_source, source_lane_type, lane_type);
  }

  const loom_amdgpu_vector_conversion_kind_flags_t kind_flags =
      loom_amdgpu_vector_conversion_kind_flags(plan->kind);
  if (iree_any_bit_set(kind_flags,
                       LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT)) {
    return loom_amdgpu_lower_vector_conversion_packed_result(
        context, source_op, plan, low_source, source_lane_type, lane_type);
  }
  if (plan->kind != LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE &&
      plan->kind < LOOM_AMDGPU_VECTOR_CONVERSION_KIND_COUNT_) {
    return loom_amdgpu_lower_vector_conversion_full_result(
        context, source_op, plan, low_source, source_lane_type, lane_type);
  }
  IREE_ASSERT_UNREACHABLE("invalid AMDGPU vector conversion plan kind");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_extract_vector_register_unit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, const loom_amdgpu_vector_extract_plan_t* plan,
    uint32_t lane_offset, uint32_t result_register_index, loom_type_t unit_type,
    loom_value_id_t* out_register_unit) {
  *out_register_unit = LOOM_VALUE_ID_INVALID;
  if (plan->lane_bit_count < 32) {
    IREE_ASSERT_TRUE(result_register_index == 0);
    const loom_amdgpu_bitfield_extract_mode_t extract_mode =
        plan->sign_extend_packed_lane
            ? LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND
            : LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_RAW_SHIFTED;
    return loom_amdgpu_extract_packed_register_lane(
        context, source_op, low_source, plan, lane_offset, extract_mode,
        unit_type, out_register_unit);
  }

  IREE_ASSERT_TRUE(plan->lane_bit_count == 32 || plan->lane_bit_count == 64);
  const uint32_t register_offset =
      lane_offset * plan->element_register_count + result_register_index;
  IREE_ASSERT(register_offset < plan->register_count);
  return loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, plan->register_count, register_offset,
      unit_type, out_register_unit);
}

static iree_status_t loom_amdgpu_lower_static_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  if (plan->lane_offset == 0 &&
      plan->result_register_count == plan->register_count &&
      !plan->sign_extend_packed_lane) {
    return loom_low_lower_bind_value(context, plan->result, low_source);
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t register_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(register_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &register_type));
  }

  loom_value_id_t registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->result_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_register_unit(
        context, source_op, low_source, plan, plan->lane_offset, i,
        register_type, &registers[i]));
  }

  return loom_amdgpu_bind_low_register_range(
      context, source_op, plan->result, registers, plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_dynamic_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  if (plan->lane_count == 1 && !plan->sign_extend_packed_lane) {
    return loom_low_lower_bind_value(context, plan->result, low_source);
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));

  loom_value_id_t selected_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_register_unit(
        context, source_op, low_source, plan, 0, register_index,
        source_lane_type, &selected_registers[register_index]));
    if (!loom_type_equal(source_lane_type, lane_type)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
          context, source_op, selected_registers[register_index],
          &selected_registers[register_index]));
    }
  }

  loom_value_id_t index_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, plan->dynamic_index, &index_lane));
  for (uint32_t i = 1; i < plan->lane_count; ++i) {
    loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, i, lane_type,
        &ordinal));

    const loom_value_id_t compare_operands[] = {
        index_lane,
        ordinal,
    };
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        compare_operands, IREE_ARRAYSIZE(compare_operands),
        loom_make_named_attr_slice(NULL, 0), &mask_lane_type, 1, &compare_op));

    const loom_value_id_t condition =
        loom_value_slice_get(loom_low_op_results(compare_op), 0);
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      loom_value_id_t table_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_register_unit(
          context, source_op, low_source, plan, i, register_index,
          source_lane_type, &table_lane));
      if (!loom_type_equal(source_lane_type, lane_type)) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
            context, source_op, table_lane, &table_lane));
      }
      const loom_value_id_t select_operands[] = {
          selected_registers[register_index],
          table_lane,
          condition,
      };
      loom_op_t* select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
          select_operands, IREE_ARRAYSIZE(select_operands),
          loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &select_op));
      selected_registers[register_index] =
          loom_value_slice_get(loom_low_op_results(select_op), 0);
    }
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             selected_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  return plan->is_dynamic ? loom_amdgpu_lower_dynamic_vector_extract(
                                context, source_op, plan)
                          : loom_amdgpu_lower_static_vector_extract(
                                context, source_op, plan);
}
