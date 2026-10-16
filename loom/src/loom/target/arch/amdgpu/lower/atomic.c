// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/atomic.h"

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/context.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/atomic_ordering.h"
#include "loom/target/arch/amdgpu/lower/atomic_subword.h"
#include "loom/target/arch/amdgpu/lower/buffer_descriptor.h"
#include "loom/target/arch/amdgpu/lower/candidates/atomic_candidates.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/memory_packet.h"
#include "loom/target/arch/amdgpu/lower/source_alloca_layout.h"
#include "loom/target/arch/amdgpu/lower/source_value_analysis.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

typedef uint32_t loom_amdgpu_atomic_rejection_flags_t;

#define LOOM_AMDGPU_ATOMIC_REJECTION_SOURCE_OP ((uint32_t)1u << 0)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND ((uint32_t)1u << 1)
#define LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE ((uint32_t)1u << 2)
#define LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE ((uint32_t)1u << 4)
#define LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND ((uint32_t)1u << 5)
#define LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE ((uint32_t)1u << 6)
#define LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT ((uint32_t)1u << 7)
#define LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING ((uint32_t)1u << 8)
#define LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE ((uint32_t)1u << 9)
#define LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING ((uint32_t)1u << 10)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE ((uint32_t)1u << 11)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE ((uint32_t)1u << 12)
#define LOOM_AMDGPU_ATOMIC_REJECTION_NATIVE_SEMANTICS ((uint32_t)1u << 13)

typedef struct loom_amdgpu_atomic_diagnostic_t {
  // Rejection bits explaining why a source atomic is not legal.
  loom_amdgpu_atomic_rejection_flags_t rejection_bits;
} loom_amdgpu_atomic_diagnostic_t;

typedef uint32_t loom_amdgpu_atomic_payload_placement_flags_t;

// Source update payload prefers VGPR storage.
#define LOOM_AMDGPU_ATOMIC_PAYLOAD_VALUE_PREFERS_VGPR ((uint32_t)1u << 0)
// Compare-exchange expected payload prefers VGPR storage.
#define LOOM_AMDGPU_ATOMIC_PAYLOAD_EXPECTED_PREFERS_VGPR ((uint32_t)1u << 1)
// Compare-exchange replacement payload prefers VGPR storage.
#define LOOM_AMDGPU_ATOMIC_PAYLOAD_REPLACEMENT_PREFERS_VGPR ((uint32_t)1u << 2)

typedef struct loom_amdgpu_atomic_source_t {
  // Generic memory-access interface for the source op.
  loom_memory_access_t access;
  // Source atomic operation family.
  loom_amdgpu_atomic_operation_kind_t operation_kind;
  // Source atomic update kind, or LOOM_AMDGPU_ATOMIC_KIND_NONE for cmpxchg.
  uint8_t atomic_kind;
  // Source update/contribution value, if the operation has one.
  loom_value_id_t value;
  // Source compare-exchange expected value, if present.
  loom_value_id_t expected;
  // Source compare-exchange replacement value, if present.
  loom_value_id_t replacement;
  // Source old-value result, if the operation returns one.
  loom_value_id_t result;
  // Source payload placement facts computed before descriptor selection.
  loom_amdgpu_atomic_payload_placement_flags_t payload_placement_flags;
} loom_amdgpu_atomic_source_t;

typedef struct loom_amdgpu_atomic_rejection_key_t {
  // Rejection bit matched by this diagnostic row.
  loom_amdgpu_atomic_rejection_flags_t rejection_bit;
  // Stable diagnostic constraint key for the matched rejection bit.
  iree_string_view_t constraint_key;
} loom_amdgpu_atomic_rejection_key_t;

typedef struct loom_amdgpu_atomic_selection_t {
  // Target-independent source memory access plan being wrapped.
  loom_low_source_memory_access_plan_t source;
  // Source atomic operation form being lowered.
  loom_amdgpu_atomic_operation_kind_t operation_kind;
  // Selected target addressing form for the atomic packet.
  loom_amdgpu_memory_address_form_t address_form;
  // Target operand path selected for each source dynamic address term.
  loom_amdgpu_memory_dynamic_index_kind_t
      dynamic_term_kinds[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Static offset value encoded in the descriptor offset immediate.
  int64_t immediate_offset;
  // Static byte offset materialized through the scalar SOFFSET operand.
  uint32_t scalar_byte_offset;
  // Stable descriptor ref selected for the active descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Mixed-bank realizations that fit u32 VADDR and eliminate dynamic SOFFSET.
  uint8_t vaddr_realization_mask;
  // Selected coherence field before module-local name interning.
  loom_amdgpu_memory_coherence_attr_t coherence_attr;
  // Explicit packets required to implement source atomic ordering.
  loom_amdgpu_atomic_ordering_selection_t ordering;
} loom_amdgpu_atomic_selection_t;

static const loom_amdgpu_atomic_rejection_key_t kAmdgpuAtomicRejectionKeys[] = {
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_NATIVE_SEMANTICS,
        .constraint_key = IREE_SVL("atomic.native_semantics"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SOURCE_OP,
        .constraint_key = IREE_SVL("atomic.source_op"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND,
        .constraint_key = IREE_SVL("atomic.operation_kind"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE,
        .constraint_key = IREE_SVL("atomic.memory_space"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE,
        .constraint_key = IREE_SVL("atomic.shape"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND,
        .constraint_key = IREE_SVL("atomic.kind"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE,
        .constraint_key = IREE_SVL("atomic.value_type"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT,
        .constraint_key = IREE_SVL("atomic.value_placement"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING,
        .constraint_key = IREE_SVL("atomic.ordering"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE,
        .constraint_key = IREE_SVL("atomic.scope"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING,
        .constraint_key = IREE_SVL("atomic.descriptor_missing"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE,
        .constraint_key = IREE_SVL("atomic.offset_immediate"),
    },
    {
        .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE,
        .constraint_key = IREE_SVL("atomic.offset_range"),
    },
};

static uint8_t loom_amdgpu_atomic_u8_attr(loom_attribute_t attr) {
  IREE_ASSERT(!loom_attr_is_absent(attr));
  return (uint8_t)loom_attr_as_i64(attr);
}

static bool loom_amdgpu_atomic_operation_kind_from_memory_access(
    loom_memory_access_operation_kind_t access_kind,
    loom_amdgpu_atomic_operation_kind_t* out_kind) {
  static const uint8_t kAmdgpuAtomicOperationKindCodeByMemoryAccess
      [LOOM_MEMORY_ACCESS_OPERATION_COUNT_] = {
          [LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_REDUCE] =
              LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE + 1u,
          [LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_RMW] =
              LOOM_AMDGPU_ATOMIC_OPERATION_RMW + 1u,
          [LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_CMPXCHG] =
              LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG + 1u,
      };
  if (access_kind >= LOOM_MEMORY_ACCESS_OPERATION_COUNT_) {
    *out_kind = LOOM_AMDGPU_ATOMIC_OPERATION_COUNT_;
    return false;
  }
  const uint8_t kind_code =
      kAmdgpuAtomicOperationKindCodeByMemoryAccess[access_kind];
  if (kind_code == 0) {
    *out_kind = LOOM_AMDGPU_ATOMIC_OPERATION_COUNT_;
    return false;
  }
  *out_kind = (loom_amdgpu_atomic_operation_kind_t)(kind_code - 1u);
  return true;
}

static void loom_amdgpu_atomic_source_describe_update(
    loom_memory_access_t access,
    loom_amdgpu_atomic_operation_kind_t operation_kind, loom_value_id_t result,
    loom_amdgpu_atomic_source_t* out_source) {
  out_source->operation_kind = operation_kind;
  out_source->atomic_kind =
      loom_amdgpu_atomic_u8_attr(loom_memory_access_atomic_kind(access));
  out_source->value = loom_memory_access_value(access);
  out_source->result = result;
}

static bool loom_amdgpu_atomic_source_describe(
    const loom_module_t* module, const loom_op_t* op,
    loom_amdgpu_atomic_source_t* out_source) {
  *out_source = (loom_amdgpu_atomic_source_t){
      .operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_COUNT_,
      .atomic_kind = LOOM_AMDGPU_ATOMIC_KIND_NONE,
      .value = LOOM_VALUE_ID_INVALID,
      .expected = LOOM_VALUE_ID_INVALID,
      .replacement = LOOM_VALUE_ID_INVALID,
      .result = LOOM_VALUE_ID_INVALID,
  };
  loom_memory_access_t access = loom_memory_access_cast(module, op);
  if (!loom_memory_access_isa(access)) {
    return false;
  }
  out_source->access = access;

  if (!loom_amdgpu_atomic_operation_kind_from_memory_access(
          loom_memory_access_operation_kind(access),
          &out_source->operation_kind)) {
    return false;
  }

  const loom_value_id_t result = op->result_count == 1
                                     ? loom_op_const_results(op)[0]
                                     : LOOM_VALUE_ID_INVALID;
  if (out_source->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG) {
    if (result == LOOM_VALUE_ID_INVALID) {
      return false;
    }
    out_source->expected = loom_memory_access_expected(access);
    out_source->replacement = loom_memory_access_replacement(access);
    out_source->result = result;
    return out_source->expected != LOOM_VALUE_ID_INVALID &&
           out_source->replacement != LOOM_VALUE_ID_INVALID;
  }

  loom_amdgpu_atomic_source_describe_update(access, out_source->operation_kind,
                                            result, out_source);
  return true;
}

static bool loom_amdgpu_atomic_prefers_global_saddr(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_memory_space_t memory_space, loom_type_t value_type) {
  // Wide global atomics avoid a four-SGPR buffer descriptor and preserve their
  // input payload instead of requiring fresh pairs for destructive returns.
  return memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
         (loom_scalar_type_bitwidth(loom_type_element_type(value_type)) == 64 ||
          loom_amdgpu_memory_cache_policy_descriptor_encoding(descriptor_set) ==
              LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH);
}

static bool loom_amdgpu_atomic_value_kind_matches(
    loom_type_t value_type, loom_amdgpu_atomic_value_kind_t value_kind) {
  switch (value_kind) {
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_B32:
      return loom_type_is_scalar(value_type) &&
             loom_scalar_type_bitwidth(loom_type_element_type(value_type)) ==
                 32;
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_B64:
      return loom_type_is_scalar(value_type) &&
             loom_scalar_type_bitwidth(loom_type_element_type(value_type)) ==
                 64;
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32:
      return loom_amdgpu_type_is_i32(value_type);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32:
      return loom_amdgpu_type_is_f32(value_type);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_F64:
      return loom_amdgpu_type_is_f64(value_type);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_I64:
      return loom_amdgpu_type_is_i64(value_type);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_PACKED_F16:
      return loom_type_is_vector(value_type) &&
             loom_type_rank(value_type) == 1 &&
             !loom_type_dim_is_dynamic_at(value_type, 0) &&
             loom_type_dim_static_size_at(value_type, 0) == 2 &&
             loom_type_element_type(value_type) == LOOM_SCALAR_TYPE_F16;
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_PACKED_BF16:
      return loom_type_is_vector(value_type) &&
             loom_type_rank(value_type) == 1 &&
             !loom_type_dim_is_dynamic_at(value_type, 0) &&
             loom_type_dim_static_size_at(value_type, 0) == 2 &&
             loom_type_element_type(value_type) == LOOM_SCALAR_TYPE_BF16;
  }
  return false;
}

static uint32_t loom_amdgpu_atomic_value_register_count(
    loom_type_t value_type) {
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(value_type));
  IREE_ASSERT_GT(element_bit_count, 0);
  uint32_t payload_bit_count = (uint32_t)element_bit_count;
  if (loom_type_is_vector(value_type)) {
    IREE_ASSERT_EQ(loom_type_rank(value_type), 1);
    IREE_ASSERT(!loom_type_dim_is_dynamic_at(value_type, 0));
    payload_bit_count *= (uint32_t)loom_type_dim_static_size_at(value_type, 0);
  } else {
    IREE_ASSERT(loom_type_is_scalar(value_type));
  }
  return (payload_bit_count + 31) / 32;
}

static bool loom_amdgpu_atomic_scalar_source_shape(
    const loom_low_source_memory_access_plan_t* source,
    uint32_t element_byte_count) {
  return source->element_byte_count == element_byte_count &&
         source->vector_lane_count == 1 &&
         source->vector_lane_byte_stride == (int64_t)element_byte_count;
}

static bool loom_amdgpu_atomic_bitwise_scalar_source_shape(
    const loom_low_source_memory_access_plan_t* source,
    loom_type_t value_type) {
  if (!loom_type_is_scalar(value_type)) {
    return false;
  }
  const int32_t bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(value_type));
  return (bit_count == 32 || bit_count == 64) &&
         loom_amdgpu_atomic_scalar_source_shape(source,
                                                (uint32_t)bit_count / 8);
}

static bool loom_amdgpu_atomic_packed_half_value_type(loom_type_t value_type) {
  return loom_type_is_vector(value_type) && loom_type_rank(value_type) == 1 &&
         !loom_type_dim_is_dynamic_at(value_type, 0) &&
         loom_type_dim_static_size_at(value_type, 0) == 2 &&
         loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_16BIT_FLOAT,
                                       loom_type_element_type(value_type));
}

bool loom_amdgpu_atomic_packed_half_source_shape(
    const loom_low_source_memory_access_plan_t* source,
    loom_type_t value_type) {
  return source->element_byte_count == 2 && source->vector_lane_count == 2 &&
         source->vector_lane_byte_stride == 2 &&
         source->vector_offset_kind ==
             LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_IDENTITY_IOTA &&
         source->minimum_alignment >= 4 &&
         loom_amdgpu_atomic_packed_half_value_type(value_type);
}

static bool loom_amdgpu_atomic_source_shape_supported(
    const loom_amdgpu_atomic_source_t* atomic_source,
    const loom_low_source_memory_access_plan_t* source,
    loom_type_t value_type) {
  if (atomic_source->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG &&
      loom_type_is_scalar(value_type) &&
      (source->element_byte_count == 1 || source->element_byte_count == 2)) {
    return loom_amdgpu_atomic_scalar_source_shape(
        source, (uint32_t)loom_scalar_type_bitwidth(
                    loom_type_element_type(value_type)) /
                    8);
  }
  if (atomic_source->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG ||
      loom_atomic_kind_is_exchange(atomic_source->atomic_kind)) {
    return loom_amdgpu_atomic_bitwise_scalar_source_shape(source, value_type);
  }
  return ((loom_amdgpu_type_is_i32(value_type) ||
           loom_amdgpu_type_is_f32(value_type)) &&
          loom_amdgpu_atomic_scalar_source_shape(source, 4)) ||
         ((loom_amdgpu_type_is_i64(value_type) ||
           loom_amdgpu_type_is_f64(value_type)) &&
          loom_amdgpu_atomic_scalar_source_shape(source, 8)) ||
         loom_amdgpu_atomic_packed_half_source_shape(source, value_type);
}

static bool loom_amdgpu_atomic_value_can_feed_vgpr(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_amdgpu_atomic_value_kind_t value_kind, bool prefers_vgpr) {
  if (prefers_vgpr) {
    return true;
  }
  const loom_type_t value_type = loom_module_value_type(module, value_id);
  switch (value_kind) {
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_B32:
      return loom_type_is_scalar(value_type) &&
             loom_scalar_type_bitwidth(loom_type_element_type(value_type)) >=
                 8 &&
             loom_scalar_type_bitwidth(loom_type_element_type(value_type)) <=
                 32;
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_B64:
      return loom_amdgpu_atomic_value_kind_matches(value_type, value_kind);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32:
      return loom_amdgpu_type_is_i32(value_type);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32:
      return loom_amdgpu_type_is_f32(value_type);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_F64:
      return loom_amdgpu_type_is_f64(value_type);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_I64:
      return loom_amdgpu_type_is_i64(value_type);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_PACKED_F16:
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_PACKED_BF16:
      return false;
  }
  return false;
}

static bool loom_amdgpu_atomic_value_can_feed_vgpr_operand(
    const loom_module_t* module,
    const loom_amdgpu_atomic_source_t* atomic_source,
    const loom_amdgpu_atomic_descriptor_candidate_t* candidate) {
  const loom_amdgpu_atomic_payload_placement_flags_t placement_flags =
      atomic_source->payload_placement_flags;
  if (atomic_source->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG) {
    return loom_amdgpu_atomic_value_can_feed_vgpr(
               module, atomic_source->expected, candidate->value_kind,
               iree_any_bit_set(
                   placement_flags,
                   LOOM_AMDGPU_ATOMIC_PAYLOAD_EXPECTED_PREFERS_VGPR)) &&
           loom_amdgpu_atomic_value_can_feed_vgpr(
               module, atomic_source->replacement, candidate->value_kind,
               iree_any_bit_set(
                   placement_flags,
                   LOOM_AMDGPU_ATOMIC_PAYLOAD_REPLACEMENT_PREFERS_VGPR));
  }
  return loom_amdgpu_atomic_value_can_feed_vgpr(
      module, atomic_source->value, candidate->value_kind,
      iree_any_bit_set(placement_flags,
                       LOOM_AMDGPU_ATOMIC_PAYLOAD_VALUE_PREFERS_VGPR));
}

static loom_amdgpu_atomic_payload_placement_flags_t
loom_amdgpu_atomic_payload_placement_from_analysis(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    const loom_amdgpu_atomic_source_t* atomic_source) {
  loom_amdgpu_atomic_payload_placement_flags_t flags = 0;
  if (atomic_source->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG) {
    if (loom_amdgpu_analyzed_source_value_prefers_vgpr(
            module, fact_table, view_regions, analysis,
            atomic_source->expected)) {
      flags |= LOOM_AMDGPU_ATOMIC_PAYLOAD_EXPECTED_PREFERS_VGPR;
    }
    if (loom_amdgpu_analyzed_source_value_prefers_vgpr(
            module, fact_table, view_regions, analysis,
            atomic_source->replacement)) {
      flags |= LOOM_AMDGPU_ATOMIC_PAYLOAD_REPLACEMENT_PREFERS_VGPR;
    }
    return flags;
  }
  if (loom_amdgpu_analyzed_source_value_prefers_vgpr(
          module, fact_table, view_regions, analysis, atomic_source->value)) {
    flags |= LOOM_AMDGPU_ATOMIC_PAYLOAD_VALUE_PREFERS_VGPR;
  }
  return flags;
}

static bool loom_amdgpu_atomic_memory_space_candidate_index(
    loom_value_fact_memory_space_t memory_space, uint32_t* out_index) {
  *out_index = 0;
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      *out_index = 0;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      *out_index = 1;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      *out_index = 2;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_atomic_address_form_candidate_index(
    loom_amdgpu_memory_address_form_t address_form, uint32_t* out_index) {
  *out_index = 0;
  switch (address_form) {
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT:
      *out_index = 0;
      return true;
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR:
      *out_index = 1;
      return true;
    case LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT:
      *out_index = 2;
      return true;
    default:
      return false;
  }
}

static iree_host_size_t loom_amdgpu_atomic_address_form_order(
    loom_value_fact_memory_space_t memory_space, bool prefer_global_saddr,
    loom_amdgpu_memory_address_form_t* out_address_forms) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      out_address_forms[0] = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
      return 1;
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      out_address_forms[0] = prefer_global_saddr
                                 ? LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR
                                 : LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
      out_address_forms[1] = prefer_global_saddr
                                 ? LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT
                                 : LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR;
      return 2;
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      out_address_forms[0] = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT;
      return 1;
    default:
      return 0;
  }
}

static bool loom_amdgpu_atomic_kind_candidate_index(
    loom_amdgpu_atomic_operation_kind_t operation_kind, uint8_t atomic_kind,
    uint32_t* out_index) {
  if (operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG) {
    *out_index = LOOM_AMDGPU_ATOMIC_KIND_INDEX_NONE;
    return true;
  }
  if (atomic_kind >= LOOM_ATOMIC_KIND_COUNT_) {
    *out_index = 0;
    return false;
  }
  *out_index = atomic_kind;
  return true;
}

static const loom_amdgpu_atomic_descriptor_candidate_range_t*
loom_amdgpu_atomic_descriptor_candidate_range(
    uint32_t memory_space_index, uint32_t address_form_index,
    loom_amdgpu_atomic_operation_kind_t operation_kind,
    uint32_t atomic_kind_index) {
  const uint32_t range_index =
      LOOM_AMDGPU_ATOMIC_DESCRIPTOR_CANDIDATE_RANGE_INDEX(
          memory_space_index, address_form_index, operation_kind,
          atomic_kind_index);
  return &kLoomAmdgpuAtomicDescriptorCandidateRanges[range_index];
}

// Source global space does not establish residency or memory granularity.
// Native floating updates must work for every allocation the source permits;
// bitwise exchange/CAS have no floating arithmetic or denormal restrictions.
static bool loom_amdgpu_atomic_native_semantics_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_atomic_operation_kind_t operation_kind, uint8_t atomic_kind,
    uint8_t scope, loom_memory_access_flags_t access_flags,
    loom_type_t value_type) {
  const bool noftz =
      iree_any_bit_set(access_flags, LOOM_MEMORY_ACCESS_FLAG_NOFTZ);
  if (operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG ||
      loom_atomic_kind_is_exchange(atomic_kind) ||
      !loom_atomic_kind_accepts_float(atomic_kind)) {
    return true;
  }
  if (atomic_kind == LOOM_ATOMIC_KIND_MINIMUMF ||
      atomic_kind == LOOM_ATOMIC_KIND_MAXIMUMF) {
    return false;
  }
  const loom_amdgpu_descriptor_set_info_t* info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  loom_amdgpu_descriptor_set_info_flags_t required = 0;
  if (atomic_kind == LOOM_ATOMIC_KIND_MINNUMF ||
      atomic_kind == LOOM_ATOMIC_KIND_MAXNUMF) {
    required |= LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_ATOMIC_NUMBER_EXTREMA;
  }
  if (memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    required |=
        scope == LOOM_ATOMIC_SCOPE_SYSTEM
            ? LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_ATOMIC_FLOAT_SYSTEM_MEMORY
            : LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_ATOMIC_FLOAT_AGENT_MEMORY;
    if (atomic_kind == LOOM_ATOMIC_KIND_ADDF && noftz &&
        loom_type_element_type(value_type) == LOOM_SCALAR_TYPE_F32) {
      required |= LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_ATOMIC_F32_ADD_DENORMALS;
    }
  }
  return iree_all_bits_set(info->flags, required);
}

bool loom_amdgpu_atomic_has_native_candidate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_atomic_operation_kind_t operation_kind, uint8_t atomic_kind,
    uint8_t scope, loom_memory_access_flags_t access_flags,
    loom_type_t value_type) {
  if (!loom_amdgpu_atomic_native_semantics_supported(
          descriptor_set, memory_space, operation_kind, atomic_kind, scope,
          access_flags, value_type)) {
    return false;
  }
  uint32_t memory_space_index = 0;
  uint32_t atomic_kind_index = 0;
  if (!loom_amdgpu_atomic_memory_space_candidate_index(memory_space,
                                                       &memory_space_index) ||
      operation_kind >= LOOM_AMDGPU_ATOMIC_OPERATION_COUNT_ ||
      !loom_amdgpu_atomic_kind_candidate_index(operation_kind, atomic_kind,
                                               &atomic_kind_index)) {
    return false;
  }

  loom_amdgpu_memory_address_form_t address_forms[2] = {0};
  const iree_host_size_t address_form_count =
      loom_amdgpu_atomic_address_form_order(
          memory_space, /*prefer_global_saddr=*/false, address_forms);
  for (iree_host_size_t address_form_ordinal = 0;
       address_form_ordinal < address_form_count; ++address_form_ordinal) {
    uint32_t address_form_index = 0;
    if (!loom_amdgpu_atomic_address_form_candidate_index(
            address_forms[address_form_ordinal], &address_form_index)) {
      continue;
    }
    const loom_amdgpu_atomic_descriptor_candidate_range_t* range =
        loom_amdgpu_atomic_descriptor_candidate_range(
            memory_space_index, address_form_index, operation_kind,
            atomic_kind_index);
    const iree_host_size_t end_candidate =
        range->first_candidate + range->candidate_count;
    for (iree_host_size_t i = range->first_candidate; i < end_candidate; ++i) {
      const loom_amdgpu_atomic_descriptor_candidate_t* candidate =
          &kLoomAmdgpuAtomicDescriptorCandidates[i];
      if (loom_amdgpu_atomic_value_kind_matches(value_type,
                                                candidate->value_kind) &&
          loom_amdgpu_descriptor_ref_ordinal(descriptor_set,
                                             candidate->descriptor_ref) !=
              LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_amdgpu_atomic_select_descriptor(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_atomic_source_t* atomic_source,
    loom_amdgpu_atomic_selection_t* selection, loom_type_t value_type,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  selection->descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_atomic_native_semantics_supported(
          descriptor_set, selection->source.memory_space,
          selection->operation_kind, atomic_source->atomic_kind,
          selection->source.atomic.scope, selection->source.access_flags,
          value_type)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_NATIVE_SEMANTICS;
    return false;
  }
  // A subword global access uses a full flat pointer: the logical buffer end
  // may bisect the physical word used by its masked compare-exchange.
  const loom_value_fact_memory_space_t descriptor_memory_space =
      selection->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT
          ? LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC
          : selection->source.memory_space;
  const bool prefer_global_saddr = loom_amdgpu_atomic_prefers_global_saddr(
      descriptor_set, descriptor_memory_space, value_type);
  bool found_kind = false;
  bool found_type = false;
  uint32_t memory_space_index = 0;
  uint32_t atomic_kind_index = 0;
  if (!loom_amdgpu_atomic_memory_space_candidate_index(descriptor_memory_space,
                                                       &memory_space_index) ||
      selection->operation_kind >= LOOM_AMDGPU_ATOMIC_OPERATION_COUNT_ ||
      !loom_amdgpu_atomic_kind_candidate_index(selection->operation_kind,
                                               atomic_source->atomic_kind,
                                               &atomic_kind_index)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND;
    return false;
  }

  loom_amdgpu_memory_address_form_t address_forms[2] = {0};
  const iree_host_size_t address_form_count =
      loom_amdgpu_atomic_address_form_order(descriptor_memory_space,
                                            prefer_global_saddr, address_forms);
  for (iree_host_size_t address_form_ordinal = 0;
       address_form_ordinal < address_form_count; ++address_form_ordinal) {
    const loom_amdgpu_memory_address_form_t address_form =
        address_forms[address_form_ordinal];
    uint32_t address_form_index = 0;
    if (!loom_amdgpu_atomic_address_form_candidate_index(address_form,
                                                         &address_form_index)) {
      continue;
    }
    const loom_amdgpu_atomic_descriptor_candidate_range_t* range =
        loom_amdgpu_atomic_descriptor_candidate_range(
            memory_space_index, address_form_index, selection->operation_kind,
            atomic_kind_index);
    if (range->candidate_count == 0) {
      continue;
    }
    found_kind = true;
    const iree_host_size_t first_candidate = range->first_candidate;
    const iree_host_size_t end_candidate =
        first_candidate + range->candidate_count;
    for (iree_host_size_t i = first_candidate; i < end_candidate; ++i) {
      const loom_amdgpu_atomic_descriptor_candidate_t* candidate =
          &kLoomAmdgpuAtomicDescriptorCandidates[i];
      if (!loom_amdgpu_atomic_value_kind_matches(value_type,
                                                 candidate->value_kind)) {
        continue;
      }
      found_type = true;
      if (!loom_amdgpu_atomic_value_can_feed_vgpr_operand(module, atomic_source,
                                                          candidate)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT;
        return false;
      }
      const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
          descriptor_set, candidate->descriptor_ref);
      if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
        continue;
      }
      selection->address_form = address_form;
      selection->descriptor_ref = candidate->descriptor_ref;
      return true;
    }
  }
  if (found_type) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
  } else {
    diagnostic->rejection_bits |=
        found_kind ? LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE
                   : LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND;
  }
  return false;
}

static bool loom_amdgpu_atomic_uses_buffer_resource(
    const loom_amdgpu_atomic_plan_t* plan) {
  return plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
         plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
}

static bool loom_amdgpu_atomic_selection_uses_buffer_resource(
    const loom_amdgpu_atomic_selection_t* selection) {
  return selection->source.memory_space ==
             LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
         selection->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
}

static bool loom_amdgpu_atomic_uses_flat_address(
    const loom_amdgpu_atomic_plan_t* plan) {
  return plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT;
}

static void loom_amdgpu_atomic_append_saddr(
    const loom_amdgpu_atomic_plan_t* plan, loom_value_id_t low_saddr,
    loom_value_id_t* operands, iree_host_size_t* operand_count) {
  if (plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    operands[(*operand_count)++] = low_saddr;
  }
}

static bool loom_amdgpu_atomic_select_offset(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_atomic_selection_t* selection,
    loom_amdgpu_memory_access_diagnostic_t* memory_diagnostic,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      descriptor_set, selection->descriptor_ref);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  if (selection->source.element_byte_count *
          selection->source.vector_lane_count <
      4) {
    // Subword CAS aligns the complete physical address before applying its
    // byte mask. Keep the descriptor offset zero and incorporate the logical
    // offset before aligning the LDS or full flat address.
    if (selection->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT &&
        !loom_amdgpu_source_memory_offset_fits_u32(
            &selection->source, selection->source.static_byte_offset)) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    selection->immediate_offset = 0;
    selection->scalar_byte_offset = 0;
    return true;
  }
  if (selection->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
    if (!loom_amdgpu_memory_access_select_flat_offset(
            descriptor_set, descriptor_ordinal, access, memory_diagnostic)) {
      return false;
    }
    selection->immediate_offset = access->immediate_offset;
    selection->scalar_byte_offset = 0;
    return true;
  }
  loom_low_immediate_kind_t expected_kind = LOOM_LOW_IMMEDIATE_KIND_UNSIGNED;
  if (selection->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    expected_kind = LOOM_LOW_IMMEDIATE_KIND_SIGNED;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, expected_kind, &offset_info) ||
      offset_info.unit_byte_count == 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE;
    return false;
  }
  if (expected_kind == LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    const int64_t signed_max = offset_info.unsigned_max > INT64_MAX
                                   ? INT64_MAX
                                   : (int64_t)offset_info.unsigned_max;
    if (offset_info.unit_byte_count != 1 ||
        selection->source.static_byte_offset < offset_info.signed_min ||
        selection->source.static_byte_offset > signed_max) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    selection->immediate_offset = selection->source.static_byte_offset;
    selection->scalar_byte_offset = 0;
    if (!loom_amdgpu_source_memory_offset_fits_u32(&selection->source,
                                                   /*static_byte_offset=*/0)) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    return true;
  }
  if (selection->source.static_byte_offset < 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  const uint64_t static_byte_offset =
      (uint64_t)selection->source.static_byte_offset;
  if (loom_amdgpu_atomic_selection_uses_buffer_resource(selection)) {
    if (offset_info.unit_byte_count != 1) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    const uint64_t immediate_offset =
        iree_min(static_byte_offset, offset_info.unsigned_max);
    const uint64_t scalar_byte_offset = static_byte_offset - immediate_offset;
    if (scalar_byte_offset > UINT32_MAX) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    selection->immediate_offset = (int64_t)immediate_offset;
    selection->scalar_byte_offset = (uint32_t)scalar_byte_offset;
    if (!loom_amdgpu_source_memory_offset_fits_u32(
            &selection->source, selection->source.static_byte_offset)) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    return true;
  }
  if ((static_byte_offset % offset_info.unit_byte_count) != 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  const uint64_t encoded_offset =
      static_byte_offset / offset_info.unit_byte_count;
  if (encoded_offset > offset_info.unsigned_max) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  selection->immediate_offset = (int64_t)encoded_offset;
  selection->scalar_byte_offset = 0;
  if (!loom_amdgpu_source_memory_offset_fits_u32(
          &selection->source, selection->source.static_byte_offset)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  return true;
}

static bool loom_amdgpu_atomic_select(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_func_like_t source_function,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    const loom_amdgpu_source_alloca_layout_t* alloca_layout,
    const loom_amdgpu_atomic_source_t* atomic_source,
    loom_amdgpu_atomic_selection_t* out_selection,
    loom_low_source_memory_access_diagnostic_t* source_diagnostic,
    loom_amdgpu_memory_access_diagnostic_t* memory_diagnostic,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  const loom_op_t* source_op = atomic_source->access.op;
  *out_selection = (loom_amdgpu_atomic_selection_t){
      .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  *source_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  *memory_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};
  *diagnostic = (loom_amdgpu_atomic_diagnostic_t){0};

  if (!loom_low_source_memory_access_plan_build(
          view_regions, source_op, /*axis_byte_strides=*/NULL,
          &out_selection->source, source_diagnostic)) {
    return false;
  }
  loom_amdgpu_atomic_operation_kind_t memory_operation_kind =
      LOOM_AMDGPU_ATOMIC_OPERATION_COUNT_;
  if (!loom_amdgpu_atomic_operation_kind_from_memory_access(
          out_selection->source.operation_kind, &memory_operation_kind) ||
      memory_operation_kind != atomic_source->operation_kind) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND;
    return false;
  }
  out_selection->operation_kind = memory_operation_kind;
  switch (out_selection->source.memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      out_selection->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
      uint64_t root_byte_offset = 0;
      if (!loom_amdgpu_source_alloca_layout_lookup_byte_offset(
              alloca_layout, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
              out_selection->source.root_value_id, &root_byte_offset)) {
        memory_diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT;
        return false;
      }
      break;
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      out_selection->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
      break;
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      out_selection->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT;
      break;
    default:
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE;
      return false;
  }
  const loom_type_t value_type =
      out_selection->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG
          ? loom_module_value_type(module, atomic_source->result)
          : loom_module_value_type(module, atomic_source->value);
  if (!loom_amdgpu_atomic_source_shape_supported(
          atomic_source, &out_selection->source, value_type)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE;
    return false;
  }
  const uint32_t packet_byte_count = out_selection->source.element_byte_count *
                                     out_selection->source.vector_lane_count;
  const loom_type_t packet_type = packet_byte_count < 4
                                      ? loom_type_scalar(LOOM_SCALAR_TYPE_I32)
                                      : value_type;
  if (!loom_amdgpu_atomic_orderings_supported(descriptor_set,
                                              &out_selection->source)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING;
    return false;
  }
  if (!loom_amdgpu_atomic_scope_supported(
          descriptor_set, &out_selection->source, packet_type)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE;
    return false;
  }

  if (packet_byte_count < 4 && out_selection->source.memory_space ==
                                   LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL) {
    out_selection->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT;
  }
  loom_amdgpu_memory_access_t memory_access =
      loom_amdgpu_memory_access_from_source(&out_selection->source);
  memory_access.address_form = out_selection->address_form;
  if (out_selection->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
    if (!loom_amdgpu_memory_access_select_flat_address(
            module, &out_selection->source, &memory_access,
            memory_diagnostic)) {
      return false;
    }
  } else if (out_selection->source.memory_space ==
             LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (!loom_amdgpu_memory_access_select_vaddr_dynamic_terms(
            module, &memory_access, memory_diagnostic)) {
      return false;
    }
  } else if (!loom_amdgpu_memory_access_select_dynamic_term_kinds(
                 module, fact_table, view_regions, analysis, &memory_access,
                 memory_diagnostic)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY;
       ++i) {
    out_selection->dynamic_term_kinds[i] = memory_access.dynamic_term_kinds[i];
  }

  if (!loom_amdgpu_atomic_select_descriptor(module, fact_table, descriptor_set,
                                            atomic_source, out_selection,
                                            packet_type, diagnostic)) {
    return false;
  }
  out_selection->coherence_attr = loom_amdgpu_atomic_select_packet_attr(
      descriptor_set, &out_selection->source);
  if (!loom_amdgpu_atomic_select_ordering(
          descriptor_set, &out_selection->source, out_selection->operation_kind,
          &out_selection->ordering)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  if (!loom_amdgpu_atomic_select_offset(descriptor_set, &memory_access,
                                        out_selection, memory_diagnostic,
                                        diagnostic)) {
    return false;
  }
  memory_access.address_form = out_selection->address_form;
  memory_access.scalar_byte_offset = out_selection->scalar_byte_offset;
  loom_amdgpu_memory_access_select_vaddr_realizations(&memory_access);
  out_selection->vaddr_realization_mask = memory_access.vaddr_realization_mask;
  return true;
}

static iree_status_t loom_amdgpu_atomic_resolve_selection(
    loom_low_lower_context_t* context,
    const loom_amdgpu_atomic_selection_t* selection,
    loom_amdgpu_atomic_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_atomic_plan_t){
      .source = selection->source,
      .operation_kind = selection->operation_kind,
      .address_form = selection->address_form,
      .immediate_offset = selection->immediate_offset,
      .scalar_byte_offset = selection->scalar_byte_offset,
      .vaddr_realization_mask = selection->vaddr_realization_mask,
      .coherence_attr =
          {
              .name_id = LOOM_STRING_ID_INVALID,
              .value = selection->coherence_attr.value,
          },
  };
  for (iree_host_size_t i = 0; i < LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY;
       ++i) {
    out_plan->dynamic_term_kinds[i] = selection->dynamic_term_kinds[i];
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_plan_memory_dynamic_terms(
      context, &out_plan->source, &out_plan->dynamic_term_plans));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, selection->descriptor_ref, &out_plan->descriptor));
  if (!iree_string_view_is_empty(selection->coherence_attr.name)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context,
                                            selection->coherence_attr.name,
                                            &out_plan->coherence_attr.name_id));
  }
  return loom_amdgpu_atomic_resolve_ordering_selection(
      context, &selection->ordering, &out_plan->ordering);
}

static iree_string_view_t loom_amdgpu_atomic_rejection_key(
    loom_amdgpu_atomic_rejection_flags_t rejection_bits) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kAmdgpuAtomicRejectionKeys);
       ++i) {
    const loom_amdgpu_atomic_rejection_key_t* row =
        &kAmdgpuAtomicRejectionKeys[i];
    if (iree_any_bit_set(rejection_bits, row->rejection_bit)) {
      return row->constraint_key;
    }
  }
  return IREE_SV("atomic.representability");
}

static uint32_t loom_amdgpu_atomic_source_payload_register_count(
    const loom_low_source_memory_access_plan_t* source) {
  const uint32_t packet_byte_count =
      source->element_byte_count * source->vector_lane_count;
  return (packet_byte_count + 3) / 4;
}

static iree_status_t loom_amdgpu_plan_atomic_constant_payloads(
    loom_low_lower_context_t* context,
    const loom_amdgpu_atomic_source_t* source,
    const loom_amdgpu_atomic_constant_payloads_t** out_payloads) {
  *out_payloads = NULL;
  const bool is_cmpxchg =
      source->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG;
  const loom_value_id_t values[] = {
      is_cmpxchg ? source->expected : source->value,
      source->replacement,
  };
  const uint32_t value_count = is_cmpxchg ? 2 : 1;
  loom_amdgpu_atomic_constant_payloads_t payloads = {0};
  for (uint32_t i = 0; i < value_count; ++i) {
    const loom_type_t type = loom_module_value_type(
        loom_low_lower_context_module(context), values[i]);
    int64_t integer = 0;
    uint32_t float_bits = 0;
    if (loom_amdgpu_value_as_i32_constant(context, values[i], &integer)) {
      payloads.bits[i] = (uint32_t)integer;
    } else if (loom_amdgpu_value_as_f32_constant(context, values[i],
                                                 &float_bits)) {
      payloads.bits[i] = float_bits;
    } else if (loom_amdgpu_type_is_i64(type) &&
               loom_value_facts_as_exact_i64(
                   loom_value_fact_table_lookup(
                       loom_low_lower_context_fact_table(context), values[i]),
                   &integer)) {
      payloads.bits[i] = (uint64_t)integer;
    } else {
      continue;
    }
    payloads.operand_mask |= (uint8_t)(1u << i);
  }
  if (payloads.operand_mask == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_atomic_constant_payloads_t* retained = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*retained), (void**)&retained));
  *retained = payloads;
  *out_payloads = retained;
  return iree_ok_status();
}

static const uint64_t* loom_amdgpu_atomic_constant_payload(
    const loom_amdgpu_atomic_plan_t* plan, uint32_t operand_index) {
  if (plan->constant_payloads == NULL ||
      (plan->constant_payloads->operand_mask & (1u << operand_index)) == 0) {
    return NULL;
  }
  return &plan->constant_payloads->bits[operand_index];
}

static iree_status_t loom_amdgpu_emit_i64_constant_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low_parts[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, (uint32_t)value,
      vgpr_type, &low_parts[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      (uint32_t)(value >> 32), vgpr_type, &low_parts[1]));
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  return loom_amdgpu_build_low_register_range(context, source_op, low_parts,
                                              IREE_ARRAYSIZE(low_parts),
                                              vgpr_x2_type, out_low_value);
}

static iree_status_t loom_amdgpu_emit_atomic_constant_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t bits, uint32_t register_count, loom_value_id_t* out_low_value) {
  if (register_count == 2) {
    return loom_amdgpu_emit_i64_constant_as_vgpr(context, source_op, bits,
                                                 out_low_value);
  }
  const loom_type_t vgpr_type = loom_low_register_type(
      loom_low_lower_context_descriptor_set(context)->stable_id,
      LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                                    (uint32_t)bits, vgpr_type, out_low_value);
}

static iree_status_t loom_amdgpu_copy_atomic_value_to_fresh_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, uint32_t register_count,
    loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_GT(register_count, 0);
  IREE_ASSERT_LE(register_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  if (register_count == 1) {
    return loom_amdgpu_emit_vgpr_b32_copy(context, source_op, low_value,
                                          out_low_value);
  }
  // Slices retain their source bank; the following moves perform any broadcast
  // from scalar registers while preserving a fresh payload for tied atomics.
  const loom_type_t low_type =
      loom_module_value_type(loom_low_lower_context_module(context), low_value);
  loom_type_t unit_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, loom_low_register_type_class_id(low_type), 1, &unit_type));
  loom_value_id_t low_parts[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {0};
  for (uint32_t i = 0; i < register_count; ++i) {
    loom_value_id_t low_part = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_value, register_count, i, unit_type,
        &low_part));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
        context, source_op, low_part, &low_parts[i]));
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, register_count, &result_type));
  return loom_amdgpu_build_low_register_range(context, source_op, low_parts,
                                              register_count, result_type,
                                              out_low_value);
}

iree_status_t loom_amdgpu_select_atomic_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_atomic_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_atomic_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_amdgpu_atomic_source_t atomic_source = {0};
  if (!loom_amdgpu_atomic_source_describe(module, source_op, &atomic_source)) {
    return iree_ok_status();
  }
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  loom_amdgpu_atomic_diagnostic_t diagnostic = {0};
  loom_amdgpu_atomic_selection_t selection = {0};
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_value_analysis_for_context(context, &analysis));
  atomic_source.payload_placement_flags =
      loom_amdgpu_atomic_payload_placement_from_analysis(
          module, loom_low_lower_context_fact_table(context), view_regions,
          analysis, &atomic_source);
  const loom_amdgpu_source_alloca_layout_t* alloca_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_lower_context(
      context, &alloca_layout));
  const bool selected = loom_amdgpu_atomic_select(
      module, loom_low_lower_context_fact_table(context),
      loom_low_lower_context_source_function(context),
      loom_low_lower_context_descriptor_set(context), view_regions, analysis,
      alloca_layout, &atomic_source, &selection, &source_diagnostic,
      &memory_diagnostic, &diagnostic);
  if (!selected) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_atomic_resolve_selection(context, &selection, out_plan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_plan_atomic_constant_payloads(
      context, &atomic_source, &out_plan->constant_payloads));
  *out_selected = true;
  if (atomic_source.result != LOOM_VALUE_ID_INVALID) {
    return loom_low_lower_plan_value_type(
        context, atomic_source.result,
        loom_low_register_type(
            loom_low_lower_context_descriptor_set(context)->stable_id,
            LOOM_AMDGPU_REG_CLASS_ID_VGPR,
            loom_amdgpu_atomic_source_payload_register_count(
                &out_plan->source)));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_atomic_value_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const uint64_t* constant_bits,
    loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type = loom_module_value_type(module, source_value);
  const uint32_t register_count =
      loom_amdgpu_atomic_value_register_count(source_type);
  if (loom_amdgpu_type_is_i64(source_type)) {
    if (constant_bits != NULL) {
      return loom_amdgpu_emit_i64_constant_as_vgpr(
          context, source_op, *constant_bits, out_low_value);
    }
    return loom_amdgpu_lookup_or_materialize_vgpr_i64(
        context, source_op, source_value, out_low_value);
  }

  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class_count(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, register_count);
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }
  if (constant_bits != NULL) {
    return loom_amdgpu_emit_atomic_constant_as_vgpr(
        context, source_op, *constant_bits, register_count, out_low_value);
  }
  return loom_amdgpu_copy_atomic_value_to_fresh_vgpr(
      context, source_op, low_value, register_count, out_low_value);
}

static iree_status_t loom_amdgpu_materialize_atomic_value_as_fresh_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const uint64_t* constant_bits,
    loom_value_id_t* out_low_value) {
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_value);
  const uint32_t register_count =
      loom_amdgpu_atomic_value_register_count(source_type);
  if (constant_bits != NULL) {
    return loom_amdgpu_emit_atomic_constant_as_vgpr(
        context, source_op, *constant_bits, register_count, out_low_value);
  }
  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);
  return loom_amdgpu_copy_atomic_value_to_fresh_vgpr(
      context, source_op, low_value, register_count, out_low_value);
}

static iree_status_t loom_amdgpu_emit_atomic_cmpxchg_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_expected, loom_value_id_t low_replacement,
    loom_type_t pair_type, loom_value_id_t* out_low_pair) {
  *out_low_pair = LOOM_VALUE_ID_INVALID;
  // AMDGPU compare-exchange packets consume replacement in the low half and
  // expected in the high half, then return the observed value in the low half.
  loom_value_id_t operands[] = {low_replacement, low_expected};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), operands,
      IREE_ARRAYSIZE(operands), pair_type, source_op->location, &concat_op));
  *out_low_pair = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_atomic(loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       const loom_amdgpu_atomic_plan_t* plan) {
  loom_amdgpu_atomic_source_t atomic_source = {0};
  if (!loom_amdgpu_atomic_source_describe(
          loom_low_lower_context_module(context), source_op, &atomic_source)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU atomic source op");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_value_id_t low_resource = loom_low_lower_lookup_value(
      context, loom_low_source_memory_access_base_view_value_id(&plan->source));
  const uint32_t payload_register_count =
      loom_amdgpu_atomic_source_payload_register_count(&plan->source);
  const uint32_t packet_byte_count =
      plan->source.element_byte_count * plan->source.vector_lane_count;

  loom_amdgpu_memory_access_t access =
      loom_amdgpu_memory_access_from_source(&plan->source);
  access.dynamic_term_plans = plan->dynamic_term_plans;
  access.address_form = plan->address_form;
  access.immediate_offset = plan->immediate_offset;
  access.scalar_byte_offset = plan->scalar_byte_offset;
  access.vaddr_realization_mask = plan->vaddr_realization_mask;
  access.payload_register_count = payload_register_count;
  access.packet_byte_count = packet_byte_count;
  if (loom_amdgpu_atomic_uses_flat_address(plan)) {
    access.vaddr_static_byte_offset =
        (uint64_t)(plan->source.static_byte_offset - plan->immediate_offset);
  }
  for (iree_host_size_t i = 0; i < LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY;
       ++i) {
    access.dynamic_term_kinds[i] = plan->dynamic_term_kinds[i];
  }
  loom_amdgpu_memory_dynamic_term_sequence_t sequence = {0};
  loom_amdgpu_memory_access_resolve_dynamic_terms(context, &access, &sequence);
  const loom_value_id_t low_base_addr =
      plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
          ? low_resource
          : LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_atomic_uses_flat_address(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_vaddr(
        context, source_op, &access, &sequence, low_resource, &low_vaddr));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
        context, source_op, &access, &sequence, low_base_addr, &low_vaddr));
  }

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  // Atomic coherence owns the packet fields independently of cache hints.
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("offset"), access.immediate_offset, attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));
  if (plan->coherence_attr.name_id != LOOM_STRING_ID_INVALID) {
    attrs[attr_count++] = (loom_named_attr_t){
        .name_id = plan->coherence_attr.name_id,
        .value = loom_attr_i64(plan->coherence_attr.value),
    };
  }
  const loom_named_attr_slice_t packet_attrs =
      loom_make_named_attr_slice(attrs, attr_count);

  loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
  if (plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, &access, &sequence, low_resource, &low_saddr));
  }
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_terms(
        context, source_op, &sequence, plan->scalar_byte_offset, &low_soffset));
  }
  loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
        context, source_op, low_resource, access.source, plan->buffer_extent,
        &low_descriptor));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_pre_ordering(context, source_op,
                                                            &plan->ordering));

  if (plan->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG) {
    loom_value_id_t low_expected = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_replacement = LOOM_VALUE_ID_INVALID;
    const bool reuses_source_payload =
        atomic_source.expected == atomic_source.replacement;
    if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_atomic_value_as_fresh_vgpr(
          context, source_op, atomic_source.expected,
          loom_amdgpu_atomic_constant_payload(plan, 0), &low_expected));
      if (reuses_source_payload) {
        low_replacement = low_expected;
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_atomic_value_as_fresh_vgpr(
            context, source_op, atomic_source.replacement,
            loom_amdgpu_atomic_constant_payload(plan, 1), &low_replacement));
      }
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_atomic_value_as_vgpr(
          context, source_op, atomic_source.expected,
          loom_amdgpu_atomic_constant_payload(plan, 0), &low_expected));
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_atomic_value_as_vgpr(
          context, source_op, atomic_source.replacement,
          loom_amdgpu_atomic_constant_payload(plan, 1), &low_replacement));
    }
    loom_type_t old_type = loom_type_none();
    if (payload_register_count == 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &old_type));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
          context, payload_register_count, &old_type));
    }
    loom_value_id_t low_old = LOOM_VALUE_ID_INVALID;
    if (packet_byte_count < 4) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subword_cmpxchg(
          context, source_op, plan, low_vaddr, low_expected, low_replacement,
          packet_attrs, &low_old));
    } else if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
      loom_type_t pair_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
          context, payload_register_count * 2, &pair_type));
      if (reuses_source_payload) {
        // The packet needs two independently placeable copies when one SSA
        // value supplies both halves of its destructive compare-exchange
        // payload.
        loom_op_t* copy_op = NULL;
        IREE_RETURN_IF_ERROR(loom_low_copy_build(
            loom_low_lower_context_builder(context), low_replacement, false,
            old_type, source_op->location, &copy_op));
        low_replacement = loom_low_copy_result(copy_op);
      }
      loom_value_id_t low_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_cmpxchg_pair(
          context, source_op, low_expected, low_replacement, pair_type,
          &low_pair));
      loom_value_id_t operands[] = {low_pair, low_descriptor, low_vaddr,
                                    low_soffset};
      const loom_tied_result_t tied_result = {
          .result_index = 0,
          .operand_index = 0,
      };
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
          packet_attrs, &pair_type, 1, &tied_result, 1, source_op->location,
          &low_op));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op,
          loom_value_slice_get(loom_low_op_results(low_op), 0), 0, old_type,
          &low_old));
    } else if (plan->address_form ==
                   LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR ||
               loom_amdgpu_atomic_uses_flat_address(plan)) {
      loom_type_t pair_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
          context, payload_register_count * 2, &pair_type));
      loom_value_id_t low_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_cmpxchg_pair(
          context, source_op, low_expected, low_replacement, pair_type,
          &low_pair));
      loom_value_id_t operands[3] = {low_vaddr, low_pair};
      iree_host_size_t operand_count = 2;
      loom_amdgpu_atomic_append_saddr(plan, low_saddr, operands,
                                      &operand_count);
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->descriptor, operands, operand_count, packet_attrs,
          &old_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
          source_op->location, &low_op));
      low_old = loom_value_slice_get(loom_low_op_results(low_op), 0);
    } else {
      loom_value_id_t operands[] = {low_vaddr, low_expected, low_replacement};
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
          packet_attrs, &old_type, 1, /*tied_results=*/NULL,
          /*tied_result_count=*/0, source_op->location, &low_op));
      low_old = loom_value_slice_get(loom_low_op_results(low_op), 0);
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_post_ordering(
        context, source_op, &plan->ordering));
    return loom_low_lower_bind_value(context, atomic_source.result, low_old);
  }

  if (plan->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_RMW) {
    const loom_type_t result_type = loom_low_register_type(
        loom_low_lower_context_descriptor_set(context)->stable_id,
        LOOM_AMDGPU_REG_CLASS_ID_VGPR, payload_register_count);
    if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
      loom_value_id_t low_fresh_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_atomic_value_as_fresh_vgpr(
          context, source_op, atomic_source.value,
          loom_amdgpu_atomic_constant_payload(plan, 0), &low_fresh_value));
      loom_value_id_t operands[] = {low_fresh_value, low_descriptor, low_vaddr,
                                    low_soffset};
      const loom_tied_result_t tied_result = {
          .result_index = 0,
          .operand_index = 0,
      };
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
          packet_attrs, &result_type, 1, &tied_result, 1, source_op->location,
          &low_op));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_post_ordering(
          context, source_op, &plan->ordering));
      return loom_low_lower_bind_value(
          context, atomic_source.result,
          loom_value_slice_get(loom_low_op_results(low_op), 0));
    }

    loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_atomic_value_as_vgpr(
        context, source_op, atomic_source.value,
        loom_amdgpu_atomic_constant_payload(plan, 0), &low_value));
    loom_value_id_t operands[3] = {low_vaddr, low_value};
    iree_host_size_t operand_count = 2;
    loom_amdgpu_atomic_append_saddr(plan, low_saddr, operands, &operand_count);
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->descriptor, operands, operand_count, packet_attrs,
        &result_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &low_op));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_post_ordering(
        context, source_op, &plan->ordering));
    return loom_low_lower_bind_value(
        context, atomic_source.result,
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_atomic_value_as_vgpr(
      context, source_op, atomic_source.value,
      loom_amdgpu_atomic_constant_payload(plan, 0), &low_value));
  if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
    loom_value_id_t operands[] = {low_value, low_descriptor, low_vaddr,
                                  low_soffset};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
        packet_attrs, /*result_types=*/NULL, /*result_count=*/0,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &low_op));
    return loom_amdgpu_emit_atomic_post_ordering(context, source_op,
                                                 &plan->ordering);
  }

  loom_value_id_t operands[3] = {low_vaddr, low_value};
  iree_host_size_t operand_count = 2;
  loom_amdgpu_atomic_append_saddr(plan, low_saddr, operands, &operand_count);
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, operand_count, packet_attrs,
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &low_op));
  return loom_amdgpu_emit_atomic_post_ordering(context, source_op,
                                               &plan->ordering);
}

iree_status_t loom_amdgpu_low_legality_verify_atomic(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_target_low_legality_module(context);
  loom_amdgpu_atomic_source_t atomic_source = {0};
  if (!loom_amdgpu_atomic_source_describe(module, op, &atomic_source)) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_atomic_selection_t selection = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  loom_amdgpu_atomic_diagnostic_t diagnostic = {0};
  const loom_view_region_table_t* view_regions =
      loom_target_low_legality_view_regions(context);
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_value_analysis_for_target_low_legality(context,
                                                                &analysis));
  atomic_source.payload_placement_flags =
      loom_amdgpu_atomic_payload_placement_from_analysis(
          module, loom_target_low_legality_fact_table(context), view_regions,
          analysis, &atomic_source);
  const loom_amdgpu_source_alloca_layout_t* alloca_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_low_legality(
      context, &alloca_layout));
  const bool selected = loom_amdgpu_atomic_select(
      module, loom_target_low_legality_fact_table(context),
      loom_target_low_legality_function(context),
      loom_target_low_legality_descriptor_set(context), view_regions, analysis,
      alloca_layout, &atomic_source, &selection, &source_diagnostic,
      &memory_diagnostic, &diagnostic);
  if (selected) {
    // Atomic coherence takes priority over advisory locality hints.
    if (iree_any_bit_set(loom_target_low_legality_diagnostic_flags(context),
                         LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS) &&
        loom_amdgpu_memory_cache_policy_is_present(
            &selection.source.cache_policy)) {
      const loom_amdgpu_memory_access_t access =
          loom_amdgpu_memory_access_from_source(&selection.source);
      return loom_amdgpu_record_memory_cache_policy_diagnostic(
          context, op, loom_target_low_legality_descriptor_set(context),
          &access, LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_DROPPED,
          /*cache_attrs=*/NULL);
    }
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = IREE_SV("atomic.representability");
  if (source_diagnostic.rejection_bits != 0) {
    constraint_key = loom_low_source_memory_access_rejection_key(
        source_diagnostic.rejection_bits);
  } else if (memory_diagnostic.rejection_bits != 0) {
    bool handled = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_access_rejection_diagnostic(
        context, op, &selection.source, &memory_diagnostic, &handled));
    if (handled) {
      return iree_ok_status();
    }
    constraint_key = loom_amdgpu_memory_access_rejection_key(
        memory_diagnostic.rejection_bits);
  } else {
    constraint_key =
        loom_amdgpu_atomic_rejection_key(diagnostic.rejection_bits);
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

static void loom_amdgpu_require_atomic_payload_storage(
    loom_low_lower_context_t* context,
    const loom_amdgpu_atomic_source_t* atomic_source) {
  if (atomic_source->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG) {
    loom_low_lower_require_source_value_storage(context,
                                                atomic_source->expected);
    loom_low_lower_require_source_value_storage(context,
                                                atomic_source->replacement);
    return;
  }
  loom_low_lower_require_source_value_storage(context, atomic_source->value);
}

void loom_amdgpu_mark_atomic_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_plan_t* plan) {
  loom_amdgpu_atomic_source_t atomic_source = {0};
  if (!loom_amdgpu_atomic_source_describe(
          loom_low_lower_context_module(context), source_op, &atomic_source)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU atomic source op");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_amdgpu_mark_source_memory_plan_storage_demands(context, &plan->source);
  loom_amdgpu_require_atomic_payload_storage(context, &atomic_source);
}
