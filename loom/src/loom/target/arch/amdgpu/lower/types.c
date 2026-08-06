// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/types.h"

#include "loom/ops/vector/storage.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/util/fact_table.h"

bool loom_amdgpu_type_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

bool loom_amdgpu_type_is_i16(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I16;
}

bool loom_amdgpu_type_is_i64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I64;
}

bool loom_amdgpu_type_is_i8(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I8;
}

bool loom_amdgpu_type_is_i1(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

uint32_t loom_amdgpu_integer_scalar_type_bit_count(
    loom_scalar_type_t scalar_type) {
  if (!loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD,
                                     scalar_type)) {
    return 0;
  }
  return (uint32_t)loom_scalar_type_bitwidth(scalar_type);
}

uint32_t loom_amdgpu_type_integer_scalar_bit_count(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return 0;
  }
  return loom_amdgpu_integer_scalar_type_bit_count(
      loom_type_element_type(type));
}

bool loom_amdgpu_type_is_address_scalar(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_ADDRESS,
                                       loom_type_element_type(type));
}

bool loom_amdgpu_source_address_value_needs_64bit(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id, loom_type_t source_type) {
  if (!loom_amdgpu_type_is_address_scalar(source_type)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(source_type);
  if (fact_table == NULL || module == NULL ||
      source_value_id >= module->values.count) {
    return element_type == LOOM_SCALAR_TYPE_OFFSET;
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, source_value_id);
  if (loom_value_facts_fit_unsigned_bit_count(facts, 32)) {
    return false;
  }
  if (element_type == LOOM_SCALAR_TYPE_OFFSET) {
    return true;
  }
  return element_type == LOOM_SCALAR_TYPE_INDEX &&
         loom_value_facts_fit_unsigned_bit_count(facts, 63);
}

bool loom_amdgpu_type_is_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

bool loom_amdgpu_type_is_f64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F64;
}

bool loom_amdgpu_type_is_f16_or_bf16(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  return loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_16BIT_FLOAT,
                                       loom_type_element_type(type));
}

typedef enum loom_amdgpu_vector_storage_rule_flag_bits_e {
  // Source vector must be static rank-1 rather than any static shape.
  LOOM_AMDGPU_VECTOR_STORAGE_RULE_FLAG_RANK1_ONLY = 1u << 0,
} loom_amdgpu_vector_storage_rule_flag_bits_t;
typedef uint8_t loom_amdgpu_vector_storage_rule_flags_t;

typedef enum loom_amdgpu_vector_storage_register_count_kind_e {
  // Register count is element_count * element_register_count.
  LOOM_AMDGPU_VECTOR_STORAGE_REGISTER_COUNT_KIND_LANE_MULTIPLE = 0,
  // Register count is the 32-bit word count required for the packed payload.
  LOOM_AMDGPU_VECTOR_STORAGE_REGISTER_COUNT_KIND_PACKED_32BIT = 1,
} loom_amdgpu_vector_storage_register_count_kind_t;

typedef struct loom_amdgpu_vector_storage_rule_t {
  // Physical storage class selected for this scalar element type.
  loom_amdgpu_vector_storage_kind_t kind;
  // Maximum logical element count accepted for this storage class.
  uint32_t maximum_element_count;
  // Payload bit count occupied by one logical source element.
  uint32_t element_bit_count;
  // Number of 32-bit register units occupied by one logical element.
  uint32_t element_register_count;
  // Additional shape constraints for the source vector type.
  loom_amdgpu_vector_storage_rule_flags_t flags;
  // Register-count derivation used for this storage class.
  loom_amdgpu_vector_storage_register_count_kind_t register_count_kind;
} loom_amdgpu_vector_storage_rule_t;

#define LOOM_AMDGPU_VECTOR_STORAGE_RULE_LANE_MULTIPLE(                  \
    storage_kind_, maximum_element_count_, element_bit_count_,          \
    element_register_count_, flags_)                                    \
  {                                                                     \
      .kind = (storage_kind_),                                          \
      .maximum_element_count = (maximum_element_count_),                \
      .element_bit_count = (element_bit_count_),                        \
      .element_register_count = (element_register_count_),              \
      .flags = (flags_),                                                \
      .register_count_kind =                                            \
          LOOM_AMDGPU_VECTOR_STORAGE_REGISTER_COUNT_KIND_LANE_MULTIPLE, \
  }

#define LOOM_AMDGPU_VECTOR_STORAGE_RULE_PACKED_32BIT(                  \
    storage_kind_, maximum_element_count_, element_bit_count_, flags_) \
  {                                                                    \
      .kind = (storage_kind_),                                         \
      .maximum_element_count = (maximum_element_count_),               \
      .element_bit_count = (element_bit_count_),                       \
      .element_register_count = 1,                                     \
      .flags = (flags_),                                               \
      .register_count_kind =                                           \
          LOOM_AMDGPU_VECTOR_STORAGE_REGISTER_COUNT_KIND_PACKED_32BIT, \
  }

static const loom_amdgpu_vector_storage_rule_t
    kAmdgpuVectorStorageRules[LOOM_SCALAR_TYPE_COUNT_] = {
        [LOOM_SCALAR_TYPE_I1] = LOOM_AMDGPU_VECTOR_STORAGE_RULE_LANE_MULTIPLE(
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK,
            LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES, 1, 2,
            LOOM_AMDGPU_VECTOR_STORAGE_RULE_FLAG_RANK1_ONLY),
        [LOOM_SCALAR_TYPE_I8] = LOOM_AMDGPU_VECTOR_STORAGE_RULE_PACKED_32BIT(
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER,
            LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8,
            LOOM_AMDGPU_VECTOR_STORAGE_RULE_FLAG_RANK1_ONLY),
        [LOOM_SCALAR_TYPE_I16] = LOOM_AMDGPU_VECTOR_STORAGE_RULE_PACKED_32BIT(
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER,
            LOOM_AMDGPU_MAX_PACKED_I16_LANES, 16,
            LOOM_AMDGPU_VECTOR_STORAGE_RULE_FLAG_RANK1_ONLY),
        [LOOM_SCALAR_TYPE_I32] = LOOM_AMDGPU_VECTOR_STORAGE_RULE_LANE_MULTIPLE(
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT,
            LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES, 32, 1, 0),
        [LOOM_SCALAR_TYPE_I64] = LOOM_AMDGPU_VECTOR_STORAGE_RULE_LANE_MULTIPLE(
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT,
            LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES / 2u, 64, 2, 0),
        [LOOM_SCALAR_TYPE_F8E4M3] =
            LOOM_AMDGPU_VECTOR_STORAGE_RULE_PACKED_32BIT(
                LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_8BIT_FLOAT,
                LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8,
                LOOM_AMDGPU_VECTOR_STORAGE_RULE_FLAG_RANK1_ONLY),
        [LOOM_SCALAR_TYPE_F8E5M2] =
            LOOM_AMDGPU_VECTOR_STORAGE_RULE_PACKED_32BIT(
                LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_8BIT_FLOAT,
                LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8,
                LOOM_AMDGPU_VECTOR_STORAGE_RULE_FLAG_RANK1_ONLY),
        [LOOM_SCALAR_TYPE_F16] = LOOM_AMDGPU_VECTOR_STORAGE_RULE_PACKED_32BIT(
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT,
            LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES, 16,
            LOOM_AMDGPU_VECTOR_STORAGE_RULE_FLAG_RANK1_ONLY),
        [LOOM_SCALAR_TYPE_BF16] = LOOM_AMDGPU_VECTOR_STORAGE_RULE_PACKED_32BIT(
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT,
            LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES, 16,
            LOOM_AMDGPU_VECTOR_STORAGE_RULE_FLAG_RANK1_ONLY),
        [LOOM_SCALAR_TYPE_F32] = LOOM_AMDGPU_VECTOR_STORAGE_RULE_LANE_MULTIPLE(
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT,
            LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES, 32, 1, 0),
        [LOOM_SCALAR_TYPE_F64] = LOOM_AMDGPU_VECTOR_STORAGE_RULE_LANE_MULTIPLE(
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT,
            LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES / 2u, 64, 2, 0),
};
static_assert(IREE_ARRAYSIZE(kAmdgpuVectorStorageRules) ==
                  LOOM_SCALAR_TYPE_COUNT_,
              "AMDGPU vector storage rules out of sync with scalar types");

static const loom_amdgpu_vector_storage_kind_flags_t
    kAmdgpuVectorStorageKindFlags[LOOM_AMDGPU_VECTOR_STORAGE_KIND_COUNT_] = {
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT] =
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_ANALYZE_REGISTER_BANK,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT] =
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_ANALYZE_REGISTER_BANK,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK] =
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_SGPR_MASK,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT] =
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER] =
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_8BIT_FLOAT] =
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD,
};

loom_amdgpu_vector_storage_kind_flags_t loom_amdgpu_vector_storage_kind_flags(
    loom_amdgpu_vector_storage_kind_t kind) {
  if (kind >= LOOM_AMDGPU_VECTOR_STORAGE_KIND_COUNT_) {
    return 0;
  }
  return kAmdgpuVectorStorageKindFlags[kind];
}

static const loom_amdgpu_vector_storage_rule_t*
loom_amdgpu_vector_storage_rule_for_element_type(
    loom_scalar_type_t element_type) {
  if (element_type >= LOOM_SCALAR_TYPE_COUNT_) {
    return NULL;
  }
  const loom_amdgpu_vector_storage_rule_t* rule =
      &kAmdgpuVectorStorageRules[element_type];
  return rule->kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_NONE ? NULL : rule;
}

static bool loom_amdgpu_type_vector_storage_with_rule(
    loom_type_t type, const loom_amdgpu_vector_storage_rule_t* rule,
    loom_amdgpu_vector_storage_t* out_storage) {
  if (rule == NULL || !loom_type_is_vector(type) ||
      loom_type_element_type(type) >= LOOM_SCALAR_TYPE_COUNT_) {
    return false;
  }

  uint32_t element_count = 0;
  if (iree_any_bit_set(rule->flags,
                       LOOM_AMDGPU_VECTOR_STORAGE_RULE_FLAG_RANK1_ONLY)) {
    element_count = loom_vector_static_rank1_lane_count(
        type, loom_type_element_type(type), rule->maximum_element_count);
  } else {
    if (!loom_type_is_all_static(type)) {
      return false;
    }
    uint64_t static_element_count = 0;
    if (!loom_type_static_element_count(type, &static_element_count) ||
        static_element_count == 0 ||
        static_element_count > rule->maximum_element_count) {
      return false;
    }
    element_count = (uint32_t)static_element_count;
  }
  if (element_count == 0) {
    return false;
  }

  const uint32_t payload_bit_count = element_count * rule->element_bit_count;
  uint32_t register_count = 0;
  switch (rule->register_count_kind) {
    case LOOM_AMDGPU_VECTOR_STORAGE_REGISTER_COUNT_KIND_LANE_MULTIPLE:
      register_count = element_count * rule->element_register_count;
      break;
    case LOOM_AMDGPU_VECTOR_STORAGE_REGISTER_COUNT_KIND_PACKED_32BIT:
      register_count = (payload_bit_count + 31u) / 32u;
      break;
  }
  if (register_count == 0) {
    return false;
  }

  *out_storage = (loom_amdgpu_vector_storage_t){
      .kind = rule->kind,
      .element_type = loom_type_element_type(type),
      .element_count = element_count,
      .register_count = register_count,
      .element_register_count = rule->element_register_count,
      .element_bit_count = rule->element_bit_count,
  };
  return true;
}

bool loom_amdgpu_type_vector_storage(
    loom_type_t type, loom_amdgpu_vector_storage_t* out_storage) {
  *out_storage = (loom_amdgpu_vector_storage_t){0};
  if (!loom_type_is_vector(type)) {
    return false;
  }
  const loom_amdgpu_vector_storage_rule_t* rule =
      loom_amdgpu_vector_storage_rule_for_element_type(
          loom_type_element_type(type));
  if (rule == NULL) {
    return false;
  }
  return loom_amdgpu_type_vector_storage_with_rule(type, rule, out_storage);
}

uint32_t loom_amdgpu_static_vector_lane_count(loom_type_t type,
                                              loom_scalar_type_t element_type,
                                              uint32_t max_lane_count) {
  return loom_vector_static_rank1_lane_count(type, element_type,
                                             max_lane_count);
}

uint32_t loom_amdgpu_static_vector_register_count(
    loom_type_t type, loom_scalar_type_t element_type,
    uint32_t max_register_count) {
  if (!loom_type_is_vector(type) || !loom_type_is_all_static(type) ||
      loom_type_element_type(type) != element_type) {
    return 0;
  }
  uint64_t element_count = 0;
  if (!loom_type_static_element_count(type, &element_count) ||
      element_count < 1 || element_count > max_register_count) {
    return 0;
  }
  return (uint32_t)element_count;
}

static uint32_t loom_amdgpu_vector_lane_count(loom_type_t type,
                                              loom_scalar_type_t element_type) {
  return loom_amdgpu_static_vector_lane_count(
      type, element_type, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

static uint32_t loom_amdgpu_vector_register_count(
    loom_type_t type, loom_scalar_type_t element_type) {
  return loom_amdgpu_static_vector_register_count(
      type, element_type, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

bool loom_amdgpu_type_is_32bit_memory_payload(loom_type_t type) {
  return loom_amdgpu_type_is_i32(type) || loom_amdgpu_type_is_f32(type) ||
         loom_amdgpu_static_vector_lane_count(
             type, LOOM_SCALAR_TYPE_I32, LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES) !=
             0 ||
         loom_amdgpu_static_vector_lane_count(
             type, LOOM_SCALAR_TYPE_F32, LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES) !=
             0;
}

uint32_t loom_amdgpu_vector_32bit_lane_count(loom_type_t type) {
  const uint32_t i32_lane_count =
      loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_I32);
  return i32_lane_count != 0
             ? i32_lane_count
             : loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_F32);
}

uint32_t loom_amdgpu_vector_32bit_register_count(loom_type_t type) {
  const uint32_t i32_register_count =
      loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_I32);
  return i32_register_count != 0
             ? i32_register_count
             : loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_F32);
}

bool loom_amdgpu_static_vector_flat_register_from_indices(
    loom_type_t type, const int64_t* indices, uint32_t* out_ordinal) {
  uint32_t ordinal = 0;
  const uint8_t rank = loom_type_rank(type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    const int64_t dimension_size = loom_type_dim_static_size_at(type, axis);
    if (dimension_size < 1 || indices[axis] < 0 ||
        indices[axis] >= dimension_size ||
        ordinal >
            (UINT32_MAX - (uint32_t)indices[axis]) / (uint32_t)dimension_size) {
      return false;
    }
    ordinal = ordinal * (uint32_t)dimension_size + (uint32_t)indices[axis];
  }
  *out_ordinal = ordinal;
  return true;
}

uint32_t loom_amdgpu_vector_i32_lane_count(loom_type_t type) {
  return loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_I32);
}

uint32_t loom_amdgpu_vector_i32_register_count(loom_type_t type) {
  return loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_I32);
}

uint32_t loom_amdgpu_vector_f32_lane_count(loom_type_t type) {
  return loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_F32);
}

uint32_t loom_amdgpu_vector_f32_register_count(loom_type_t type) {
  return loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_F32);
}

uint32_t loom_amdgpu_vector_i1_lane_count(loom_type_t type) {
  return loom_amdgpu_static_vector_lane_count(
      type, LOOM_SCALAR_TYPE_I1, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

uint32_t loom_amdgpu_vector_i8_lane_count(loom_type_t type) {
  return loom_amdgpu_static_vector_lane_count(type, LOOM_SCALAR_TYPE_I8,
                                              LOOM_AMDGPU_MAX_PACKED_I8_LANES);
}

bool loom_amdgpu_type_packed_integer_storage(loom_type_t type,
                                             uint32_t* out_payload_bit_count,
                                             uint32_t* out_register_count) {
  *out_payload_bit_count = 0;
  *out_register_count = 0;
  loom_vector_packed_integer_storage_shape_t shape;
  if (!loom_vector_packed_integer_storage_shape(
          type, /*storage_unit_bit_count=*/32,
          LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS, &shape)) {
    return false;
  }
  *out_payload_bit_count = shape.payload_bit_count;
  *out_register_count = shape.storage_unit_count;
  return true;
}

static bool loom_amdgpu_type_packed_vector_storage(
    loom_type_t type, loom_amdgpu_vector_storage_kind_t expected_kind,
    uint32_t* out_payload_bit_count, uint32_t* out_register_count) {
  *out_payload_bit_count = 0;
  *out_register_count = 0;
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(type, &storage) ||
      storage.kind != expected_kind) {
    return false;
  }
  *out_payload_bit_count = storage.element_count * storage.element_bit_count;
  *out_register_count = storage.register_count;
  return true;
}

bool loom_amdgpu_type_packed_8bit_float_storage(loom_type_t type,
                                                uint32_t* out_payload_bit_count,
                                                uint32_t* out_register_count) {
  return loom_amdgpu_type_packed_vector_storage(
      type, LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_8BIT_FLOAT,
      out_payload_bit_count, out_register_count);
}

bool loom_amdgpu_type_packed_16bit_float_storage(
    loom_type_t type, uint32_t* out_payload_bit_count,
    uint32_t* out_register_count) {
  return loom_amdgpu_type_packed_vector_storage(
      type, LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT,
      out_payload_bit_count, out_register_count);
}

bool loom_amdgpu_type_is_byte_addressable_view(loom_type_t type) {
  if (!loom_type_is_view(type)) {
    return false;
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  return element_bit_count > 0 && (element_bit_count % 8) == 0;
}

bool loom_amdgpu_value_is_i32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_address_scalar(loom_low_lower_context_t* context,
                                         loom_value_id_t value_id) {
  return loom_amdgpu_type_is_address_scalar(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_f32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id) {
  return loom_amdgpu_type_is_f32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_f16_or_bf16(loom_low_lower_context_t* context,
                                      loom_value_id_t value_id) {
  return loom_amdgpu_type_is_f16_or_bf16(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_byte_addressable_view(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_byte_addressable_view(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

iree_status_t loom_amdgpu_make_sgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, out_type);
}

iree_status_t loom_amdgpu_make_sgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, LOOM_AMDGPU_REG_CLASS_ID_SGPR, unit_count, out_type);
}

iree_status_t loom_amdgpu_make_vgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, out_type);
}

iree_status_t loom_amdgpu_make_scc_type(loom_low_lower_context_t* context,
                                        loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, out_type);
}

iree_status_t loom_amdgpu_make_vcc_type(loom_low_lower_context_t* context,
                                        loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, LOOM_AMDGPU_REG_CLASS_ID_VCC, 1, out_type);
}

iree_status_t loom_amdgpu_make_vgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, LOOM_AMDGPU_REG_CLASS_ID_VGPR, unit_count, out_type);
}

bool loom_amdgpu_low_type_is_register_class(loom_low_lower_context_t* context,
                                            loom_type_t type,
                                            uint16_t reg_class_id) {
  if (!loom_low_type_is_register(type)) {
    return false;
  }
  return loom_low_register_type_descriptor_set_stable_id(type) ==
             loom_low_lower_context_descriptor_set(context)->stable_id &&
         loom_low_register_type_class_id(type) == reg_class_id;
}

bool loom_amdgpu_low_type_is_register_class_count(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    uint32_t register_unit_count) {
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_unit_count(type) != register_unit_count) {
    return false;
  }
  return loom_amdgpu_low_type_is_register_class(context, type, reg_class_id);
}

bool loom_amdgpu_low_value_is_register_class_count(
    loom_low_lower_context_t* context, loom_value_id_t low_value,
    uint16_t reg_class_id, uint32_t register_unit_count) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  return loom_amdgpu_low_type_is_register_class_count(
      context, loom_module_value_type(module, low_value), reg_class_id,
      register_unit_count);
}

loom_type_t loom_amdgpu_low_register_lane_type(const loom_module_t* module,
                                               loom_value_id_t low_value) {
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  if (!loom_low_type_is_register(low_type)) {
    return loom_type_none();
  }
  return loom_low_register_carrier_type_with_unit_count(low_type, 1);
}
