// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/mask.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

iree_status_t loom_amdgpu_emit_i1_mask_select(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_value_id_t false_value,
                                              loom_value_id_t true_value,
                                              loom_value_id_t condition,
                                              loom_type_t mask_type,
                                              loom_value_id_t* out_value) {
  loom_value_id_t changed = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64, false_value,
      true_value, mask_type, &changed));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, changed,
      condition, mask_type, &changed));
  return loom_amdgpu_emit_binary(context, source_op,
                                 LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
                                 false_value, changed, mask_type, out_value);
}

typedef enum loom_amdgpu_select_immediate_value_role_e {
  LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE = 0,
  LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE = 1,
} loom_amdgpu_select_immediate_value_role_t;

typedef enum loom_amdgpu_select_immediate_operand_kind_e {
  LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION = 0,
  LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_FALSE_LANE = 1,
  LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_TRUE_LANE = 2,
} loom_amdgpu_select_immediate_operand_kind_t;

typedef enum loom_amdgpu_select_immediate_attr_kind_e {
  LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32 = 0,
  LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_FALSE_VALUE = 1,
  LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_TRUE_VALUE = 2,
} loom_amdgpu_select_immediate_attr_kind_t;

typedef struct loom_amdgpu_select_immediate_attr_t {
  // Descriptor attribute receiving the selected source bits.
  loom_amdgpu_select_immediate_attr_kind_t kind;
  // Select source lane whose exact bits populate the attribute.
  loom_amdgpu_select_immediate_value_role_t value_role;
  // True when the attribute is encoded as an inline source constrained to
  // 0..64.
  bool requires_inline_range;
} loom_amdgpu_select_immediate_attr_t;

typedef struct loom_amdgpu_select_immediate_candidate_t {
  // Byte offset to the plan descriptor row selected by this immediate form.
  iree_host_size_t descriptor_offset;
  // Operand payloads consumed by the descriptor.
  loom_amdgpu_select_immediate_operand_kind_t operands[2];
  // Number of entries in operands.
  uint8_t operand_count;
  // Attribute payloads emitted with the descriptor.
  loom_amdgpu_select_immediate_attr_t attrs[2];
  // Number of entries in attrs.
  uint8_t attr_count;
} loom_amdgpu_select_immediate_candidate_t;

static const loom_amdgpu_select_immediate_candidate_t
    kLoomAmdgpuSelectImmediateCandidates[] = {
        {
            .descriptor_offset = offsetof(
                loom_amdgpu_vector_select_plan_t,
                cndmask_descriptors.src0_literal_src1_inline_descriptor),
            .operands = {LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION},
            .operand_count = 1,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE,
                    },
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_TRUE_VALUE,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE,
                        .requires_inline_range = true,
                    },
                },
            .attr_count = 2,
        },
        {
            .descriptor_offset = offsetof(
                loom_amdgpu_vector_select_plan_t,
                cndmask_descriptors.src1_literal_src0_inline_descriptor),
            .operands = {LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION},
            .operand_count = 1,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE,
                    },
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_FALSE_VALUE,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE,
                        .requires_inline_range = true,
                    },
                },
            .attr_count = 2,
        },
        {
            .descriptor_offset =
                offsetof(loom_amdgpu_vector_select_plan_t,
                         cndmask_descriptors.src1_inline_descriptor),
            .operands =
                {
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_FALSE_LANE,
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION,
                },
            .operand_count = 2,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_TRUE_VALUE,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE,
                        .requires_inline_range = true,
                    },
                },
            .attr_count = 1,
        },
        {
            .descriptor_offset =
                offsetof(loom_amdgpu_vector_select_plan_t,
                         cndmask_descriptors.src0_inline_descriptor),
            .operands =
                {
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_TRUE_LANE,
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION,
                },
            .operand_count = 2,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_FALSE_VALUE,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE,
                        .requires_inline_range = true,
                    },
                },
            .attr_count = 1,
        },
        {
            .descriptor_offset =
                offsetof(loom_amdgpu_vector_select_plan_t,
                         cndmask_descriptors.src0_literal_descriptor),
            .operands =
                {
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_TRUE_LANE,
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION,
                },
            .operand_count = 2,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE,
                    },
                },
            .attr_count = 1,
        },
        {
            .descriptor_offset =
                offsetof(loom_amdgpu_vector_select_plan_t,
                         cndmask_descriptors.src1_literal_descriptor),
            .operands =
                {
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_FALSE_LANE,
                    LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION,
                },
            .operand_count = 2,
            .attrs =
                {
                    {
                        .kind = LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32,
                        .value_role = LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE,
                    },
                },
            .attr_count = 1,
        },
};

static const loom_low_lower_resolved_descriptor_t*
loom_amdgpu_select_immediate_candidate_descriptor(
    const loom_amdgpu_vector_select_plan_t* plan,
    const loom_amdgpu_select_immediate_candidate_t* candidate) {
  const uint8_t* plan_bytes = (const uint8_t*)plan;
  const void* descriptor_bytes = plan_bytes + candidate->descriptor_offset;
  return (const loom_low_lower_resolved_descriptor_t*)descriptor_bytes;
}

static iree_string_view_t loom_amdgpu_select_immediate_attr_name(
    loom_amdgpu_select_immediate_attr_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_IMM32:
      return IREE_SV("imm32");
    case LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_FALSE_VALUE:
      return IREE_SV("false_value");
    case LOOM_AMDGPU_SELECT_IMMEDIATE_ATTR_TRUE_VALUE:
      return IREE_SV("true_value");
  }
  return iree_string_view_empty();
}

static bool loom_amdgpu_select_immediate_bits(
    loom_amdgpu_select_immediate_value_role_t role, bool false_is_exact,
    uint32_t false_bits, bool true_is_exact, uint32_t true_bits,
    uint32_t* out_bits) {
  switch (role) {
    case LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE:
      *out_bits = false_bits;
      return false_is_exact;
    case LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_TRUE:
      *out_bits = true_bits;
      return true_is_exact;
  }
  *out_bits = 0;
  return false;
}

static bool loom_amdgpu_select_immediate_candidate_matches(
    const loom_amdgpu_vector_select_plan_t* plan,
    const loom_amdgpu_select_immediate_candidate_t* candidate,
    bool false_is_exact, uint32_t false_bits, bool true_is_exact,
    uint32_t true_bits) {
  const loom_low_lower_resolved_descriptor_t* descriptor =
      loom_amdgpu_select_immediate_candidate_descriptor(plan, candidate);
  if (descriptor->descriptor == NULL) {
    return false;
  }
  for (uint8_t i = 0; i < candidate->attr_count; ++i) {
    uint32_t bits = 0;
    if (!loom_amdgpu_select_immediate_bits(candidate->attrs[i].value_role,
                                           false_is_exact, false_bits,
                                           true_is_exact, true_bits, &bits)) {
      return false;
    }
    if (candidate->attrs[i].requires_inline_range && bits > 64) {
      return false;
    }
  }
  return true;
}

struct loom_amdgpu_select_lane_plan_t {
  // Exact false and true payload words used by the selected immediate form.
  uint32_t bits[2];
  // Physical 32-bit register lane governed by this recipe.
  uint8_t lane;
  // One-based immediate-table row; zero reuses the equal true input lane.
  uint8_t recipe;
};

static_assert(sizeof(loom_amdgpu_select_lane_plan_t) == 12,
              "data-select lane recipes must remain compact");
static_assert(LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES <= UINT8_MAX,
              "select lane ordinals must fit the retained recipe");
static_assert(IREE_ARRAYSIZE(kLoomAmdgpuSelectImmediateCandidates) < UINT8_MAX,
              "one-based select recipe ordinals must fit in a byte");

static iree_status_t loom_amdgpu_select_data_lanes(
    loom_low_lower_context_t* context, bool allows_lane_immediates,
    loom_amdgpu_vector_select_plan_t* plan) {
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_amdgpu_select_lane_plan_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint8_t lane_count = 0;
  for (uint32_t lane = 0; lane < plan->lane_count; ++lane) {
    loom_amdgpu_select_lane_plan_t selected = {.lane = lane};
    const bool false_is_exact = loom_amdgpu_source_lane_as_u32_bits(
        fact_table, module, plan->false_value, lane, &selected.bits[0]);
    const bool true_is_exact = loom_amdgpu_source_lane_as_u32_bits(
        fact_table, module, plan->true_value, lane, &selected.bits[1]);
    if (false_is_exact && true_is_exact &&
        selected.bits[0] == selected.bits[1]) {
      lanes[lane_count++] = selected;
      continue;
    }
    if (!allows_lane_immediates) {
      continue;
    }
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(kLoomAmdgpuSelectImmediateCandidates); ++i) {
      if (loom_amdgpu_select_immediate_candidate_matches(
              plan, &kLoomAmdgpuSelectImmediateCandidates[i], false_is_exact,
              selected.bits[0], true_is_exact, selected.bits[1])) {
        selected.recipe = i + 1;
        lanes[lane_count++] = selected;
        break;
      }
    }
  }
  if (lane_count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_select_lane_plan_t* retained = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, lane_count * sizeof(*retained), (void**)&retained));
  memcpy(retained, lanes, lane_count * sizeof(*retained));
  plan->payload.data.lanes = retained;
  plan->payload.data.lane_count = lane_count;
  return iree_ok_status();
}

static bool loom_amdgpu_select_vector_storage(
    loom_type_t result_type, loom_amdgpu_vector_storage_t* out_storage,
    bool* out_full_width_storage, bool* out_allows_lane_immediates) {
  *out_full_width_storage = false;
  *out_allows_lane_immediates = false;
  if (!loom_amdgpu_type_vector_storage(result_type, out_storage)) {
    return false;
  }
  const loom_amdgpu_vector_storage_kind_flags_t storage_flags =
      loom_amdgpu_vector_storage_kind_flags(out_storage->kind);
  if (iree_any_bit_set(storage_flags,
                       LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_SGPR_MASK)) {
    return false;
  }
  const bool packed_payload_storage = iree_any_bit_set(
      storage_flags, LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD);
  const bool full_width_storage = !packed_payload_storage;
  *out_full_width_storage = full_width_storage;
  *out_allows_lane_immediates =
      full_width_storage && out_storage->element_register_count == 1;
  return true;
}

static bool loom_amdgpu_select_scalar_splat_condition(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition, uint32_t expected_lane_count,
    loom_value_id_t* out_scalar_condition) {
  *out_scalar_condition = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_vector_i1_lane_count(
          loom_module_value_type(module, condition)) != expected_lane_count) {
    return false;
  }
  loom_value_id_t scalar = LOOM_VALUE_ID_INVALID;
  if (!loom_value_fact_table_query_uniform_element_origin(fact_table, module,
                                                          condition, &scalar)) {
    return false;
  }
  if (!loom_amdgpu_type_is_i1(loom_module_value_type(module, scalar))) {
    return false;
  }
  *out_scalar_condition = scalar;
  return true;
}

static iree_status_t loom_amdgpu_publish_select_carrier(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_select_plan_t* plan) {
  const bool scalar_carrier =
      plan->payload_kind == LOOM_AMDGPU_SELECT_PAYLOAD_KIND_I1_MASK ||
      plan->payload_kind == LOOM_AMDGPU_SELECT_PAYLOAD_KIND_I1_BOOL ||
      plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC ||
      plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SGPR_BOOL;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context,
      scalar_carrier ? LOOM_AMDGPU_REG_CLASS_ID_SGPR
                     : LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      plan->lane_count, &result_type));
  return loom_low_lower_plan_value_type(context, plan->result, result_type);
}

iree_status_t loom_amdgpu_select_vector_select_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_select_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_select_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_amdgpu_vector_storage_t storage = {0};
  bool full_width_storage = false;
  bool allows_lane_immediates = false;
  if (!loom_amdgpu_select_vector_storage(result_type, &storage,
                                         &full_width_storage,
                                         &allows_lane_immediates)) {
    return iree_ok_status();
  }
  const loom_value_id_t condition = loom_vector_select_condition(source_op);
  const loom_value_id_t true_value = loom_vector_select_true_value(source_op);
  const loom_value_id_t false_value = loom_vector_select_false_value(source_op);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (!loom_type_equal(loom_module_value_type(module, true_value),
                       result_type) ||
      !loom_type_equal(loom_module_value_type(module, false_value),
                       result_type)) {
    return iree_ok_status();
  }

  loom_amdgpu_select_condition_kind_t condition_kind =
      LOOM_AMDGPU_SELECT_CONDITION_KIND_NONE;
  loom_value_id_t selected_condition = condition;
  uint32_t registers_per_condition_lane = 1;
  const uint32_t condition_lane_count = loom_amdgpu_vector_i1_lane_count(
      loom_module_value_type(module, condition));
  if (full_width_storage && condition_lane_count == storage.element_count) {
    condition_kind = LOOM_AMDGPU_SELECT_CONDITION_KIND_VECTOR_MASK;
    registers_per_condition_lane = storage.element_register_count;
  } else if (loom_amdgpu_select_scalar_splat_condition(
                 module, fact_table, condition, storage.element_count,
                 &selected_condition)) {
    condition_kind = LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK;
  } else if (condition_lane_count == storage.element_count) {
    condition_kind = LOOM_AMDGPU_SELECT_CONDITION_KIND_VECTOR_MASK;
    selected_condition = condition;
  } else {
    return iree_ok_status();
  }

  const bool packed_mask =
      !full_width_storage &&
      condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_VECTOR_MASK;
  loom_amdgpu_cndmask_b32_descriptors_t cndmask_descriptors = {0};
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_cndmask_b32_descriptors(
      context, LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_REGISTER,
      packed_mask ? 0 : LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_ALL,
      &cndmask_descriptors, &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_vector_select_plan_t){
      .payload_kind = LOOM_AMDGPU_SELECT_PAYLOAD_KIND_DATA,
      .condition_kind = condition_kind,
      .cndmask_descriptors = cndmask_descriptors,
      .condition = selected_condition,
      .true_value = true_value,
      .false_value = false_value,
      .result = result,
      .lane_count = storage.register_count,
      .registers_per_condition_lane = registers_per_condition_lane,
  };
  if (packed_mask) {
    out_plan->payload_kind = LOOM_AMDGPU_SELECT_PAYLOAD_KIND_PACKED_DATA;
    out_plan->payload.packed = (loom_amdgpu_packed_select_payload_t){
        .imm32_attr_name_id = LOOM_STRING_ID_INVALID,
        .element_count = storage.element_count,
        .element_bit_count = storage.element_bit_count,
    };
    bool literal_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT,
        &out_plan->payload.packed.merge_descriptor, &literal_present));
    if (!literal_present) {
      const loom_amdgpu_descriptor_resolution_t resolutions[] = {
          {LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32,
           &out_plan->payload.packed.merge_descriptor},
          {LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
           &out_plan->payload.packed.mask_constant_descriptor},
      };
      bool present = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
          context, resolutions, IREE_ARRAYSIZE(resolutions), &present));
      if (!present) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_intern(context, IREE_SV("imm32"),
                             &out_plan->payload.packed.imm32_attr_name_id));
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_data_lanes(
        context, allows_lane_immediates, out_plan));
  }
  *out_selected = true;
  return loom_amdgpu_publish_select_carrier(context, out_plan);
}

iree_status_t loom_amdgpu_low_legality_verify_vector_select(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_vector_select_isa(op) ||
      !loom_amdgpu_low_legality_bundle_is_amdgpu(
          loom_target_low_legality_bundle(context))) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t result = loom_vector_select_result(op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_amdgpu_vector_storage_t storage = {0};
  bool full_width_storage = false;
  bool allows_lane_immediates = false;
  if (!loom_amdgpu_select_vector_storage(result_type, &storage,
                                         &full_width_storage,
                                         &allows_lane_immediates)) {
    return iree_ok_status();
  }
  (void)allows_lane_immediates;

  *out_handled = true;
  if (!loom_type_equal(
          loom_module_value_type(module, loom_vector_select_true_value(op)),
          result_type) ||
      !loom_type_equal(
          loom_module_value_type(module, loom_vector_select_false_value(op)),
          result_type)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("select.payload_type"));
  }

  const loom_value_id_t condition = loom_vector_select_condition(op);
  const uint32_t condition_lane_count = loom_amdgpu_vector_i1_lane_count(
      loom_module_value_type(module, condition));
  if (condition_lane_count != storage.element_count) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("select.mask_shape"));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_select_scalar_storage(
    loom_type_t type, uint32_t* out_register_count,
    bool* out_allows_lane_immediates) {
  *out_register_count = 0;
  *out_allows_lane_immediates = false;
  if (loom_amdgpu_type_is_i32(type) ||
      loom_amdgpu_type_is_address_scalar(type) ||
      loom_amdgpu_type_is_f32(type)) {
    *out_register_count = 1;
    *out_allows_lane_immediates = true;
    return true;
  }
  if (loom_amdgpu_type_is_i64(type) || loom_amdgpu_type_is_f64(type)) {
    *out_register_count = 2;
    return true;
  }
  if (loom_type_is_scalar(type) &&
      loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_I8 |
                                        LOOM_SCALAR_TYPE_SET_I16 |
                                        LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
                                    loom_type_element_type(type))) {
    *out_register_count = 1;
    return true;
  }
  return false;
}

static bool loom_amdgpu_select_payload_storage(
    loom_type_t type, uint32_t* out_register_count,
    bool* out_allows_lane_immediates) {
  if (loom_amdgpu_select_scalar_storage(type, out_register_count,
                                        out_allows_lane_immediates)) {
    return true;
  }
  loom_amdgpu_vector_storage_t storage = {0};
  bool unused_full_width_storage = false;
  if (!loom_amdgpu_select_vector_storage(type, &storage,
                                         &unused_full_width_storage,
                                         out_allows_lane_immediates)) {
    return false;
  }
  *out_register_count = storage.register_count;
  return true;
}

static bool loom_amdgpu_select_i1_splat_constant(
    loom_low_lower_context_t* context, loom_value_id_t value,
    bool* out_constant) {
  if (loom_amdgpu_value_as_i1_constant(context, value, out_constant)) {
    return true;
  }
  *out_constant = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(loom_module_value_type(module, value),
                                       &storage) ||
      storage.kind != LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK) {
    return false;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL) {
    return false;
  }
  loom_value_fact_uniform_element_t uniform = {0};
  return loom_value_facts_query_uniform_element(
             &fact_table->context,
             loom_value_fact_table_lookup(fact_table, value), &uniform) &&
         loom_value_facts_as_exact_bool(uniform.element, out_constant);
}

static uint8_t loom_amdgpu_select_mask_operand_kind(
    loom_low_lower_context_t* context, loom_value_id_t value) {
  bool constant = false;
  if (!loom_amdgpu_select_i1_splat_constant(context, value, &constant)) {
    return LOOM_AMDGPU_MASK_OPERAND_VALUE;
  }
  return constant ? LOOM_AMDGPU_MASK_OPERAND_EXEC
                  : LOOM_AMDGPU_MASK_OPERAND_ZERO;
}

static loom_amdgpu_mask_select_recipe_t loom_amdgpu_select_i1_mask_recipe(
    const loom_amdgpu_vector_select_plan_t* plan) {
  if (plan->true_value == plan->false_value) {
    return LOOM_AMDGPU_MASK_SELECT_TRUE_VALUE;
  }
  if (plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC ||
      plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SGPR_BOOL) {
    return LOOM_AMDGPU_MASK_SELECT_UNIFORM;
  }
  const uint8_t true_kind = plan->payload.mask.operands.true_value;
  const uint8_t false_kind = plan->payload.mask.operands.false_value;
  if (true_kind != LOOM_AMDGPU_MASK_OPERAND_VALUE &&
      false_kind != LOOM_AMDGPU_MASK_OPERAND_VALUE) {
    if (true_kind == false_kind) {
      return LOOM_AMDGPU_MASK_SELECT_TRUE_VALUE;
    }
    return true_kind == LOOM_AMDGPU_MASK_OPERAND_EXEC
               ? LOOM_AMDGPU_MASK_SELECT_CONDITION
               : LOOM_AMDGPU_MASK_SELECT_INVERSE_CONDITION;
  }
  if (plan->true_value == plan->condition ||
      true_kind == LOOM_AMDGPU_MASK_OPERAND_EXEC) {
    return LOOM_AMDGPU_MASK_SELECT_CONDITION_OR_FALSE;
  }
  if (true_kind == LOOM_AMDGPU_MASK_OPERAND_ZERO) {
    return LOOM_AMDGPU_MASK_SELECT_INVERSE_AND_FALSE;
  }
  if (false_kind == LOOM_AMDGPU_MASK_OPERAND_EXEC) {
    return LOOM_AMDGPU_MASK_SELECT_TRUE_OR_INVERSE;
  }
  if (plan->false_value == plan->condition ||
      false_kind == LOOM_AMDGPU_MASK_OPERAND_ZERO) {
    return LOOM_AMDGPU_MASK_SELECT_CONDITION_AND_TRUE;
  }
  return LOOM_AMDGPU_MASK_SELECT_MERGE;
}

static iree_status_t loom_amdgpu_resolve_i1_mask_select_descriptors(
    loom_low_lower_context_t* context, loom_amdgpu_vector_select_plan_t* plan,
    bool* out_present) {
  *out_present = false;
  bool all_present = true;
  bool scc_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32, &plan->scc_descriptor,
      &scc_present));
  all_present = all_present && scc_present;
  if (plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SGPR_BOOL) {
    bool compare_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32_SRC1_INLINE,
        &plan->sgpr_bool_compare_descriptor, &compare_present));
    all_present = all_present && compare_present;
  }
  bool exec_read_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      &plan->payload.mask.exec_read_descriptor, &exec_read_present));
  all_present = all_present && exec_read_present;
  if (!all_present) {
    return iree_ok_status();
  }

  enum {
    MASK_AND = 1u << 0,
    MASK_OR = 1u << 1,
    MASK_XOR = 1u << 2,
  };
  uint8_t descriptor_bits = 0;
  switch ((loom_amdgpu_mask_select_recipe_t)plan->payload.mask.recipe) {
    case LOOM_AMDGPU_MASK_SELECT_TRUE_VALUE:
    case LOOM_AMDGPU_MASK_SELECT_UNIFORM:
    case LOOM_AMDGPU_MASK_SELECT_CONDITION:
      break;
    case LOOM_AMDGPU_MASK_SELECT_INVERSE_CONDITION:
      descriptor_bits = MASK_XOR;
      break;
    case LOOM_AMDGPU_MASK_SELECT_CONDITION_OR_FALSE:
      descriptor_bits = MASK_OR;
      break;
    case LOOM_AMDGPU_MASK_SELECT_INVERSE_AND_FALSE:
      descriptor_bits = MASK_AND | MASK_XOR;
      break;
    case LOOM_AMDGPU_MASK_SELECT_TRUE_OR_INVERSE:
      descriptor_bits = MASK_OR | MASK_XOR;
      break;
    case LOOM_AMDGPU_MASK_SELECT_CONDITION_AND_TRUE:
      descriptor_bits = MASK_AND;
      break;
    case LOOM_AMDGPU_MASK_SELECT_MERGE:
      descriptor_bits = MASK_AND | MASK_OR | MASK_XOR;
      break;
  }
  const loom_amdgpu_descriptor_resolution_t resolutions[] = {
      {LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
       &plan->payload.mask.and_descriptor},
      {LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64, &plan->payload.mask.or_descriptor},
      {LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
       &plan->payload.mask.xor_descriptor},
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(resolutions); ++i) {
    if (iree_any_bit_set(descriptor_bits, 1u << i)) {
      bool present = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
          context, resolutions[i].descriptor_ref, resolutions[i].out_descriptor,
          &present));
      all_present = all_present && present;
    }
  }
  *out_present = all_present;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_scf_select_i1_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_scf_select_result(source_op);
  const loom_value_id_t condition = loom_scf_select_condition(source_op);
  const loom_value_id_t true_value = loom_scf_select_true_value(source_op);
  const loom_value_id_t false_value = loom_scf_select_false_value(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  uint32_t register_count = 0;
  if (loom_amdgpu_type_is_i1(result_type)) {
    register_count = 2;
  } else {
    loom_amdgpu_vector_storage_t storage = {0};
    if (!loom_amdgpu_type_vector_storage(result_type, &storage) ||
        storage.kind != LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK) {
      return iree_ok_status();
    }
    register_count = storage.register_count;
  }
  if (!loom_amdgpu_type_is_i1(loom_module_value_type(module, condition)) ||
      !loom_type_equal(loom_module_value_type(module, true_value),
                       result_type) ||
      !loom_type_equal(loom_module_value_type(module, false_value),
                       result_type)) {
    return iree_ok_status();
  }

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  if (loom_amdgpu_type_is_i1(result_type) &&
      loom_amdgpu_low_type_is_register_class_count(
          context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1)) {
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32,
        &out_plan->scc_descriptor, &present));
    if (!present) {
      return iree_ok_status();
    }
    out_plan->payload_kind = LOOM_AMDGPU_SELECT_PAYLOAD_KIND_I1_BOOL;
    out_plan->condition = condition;
    out_plan->true_value = true_value;
    out_plan->false_value = false_value;
    out_plan->result = result;
    out_plan->lane_count = 1;
    *out_selected = true;
    return loom_amdgpu_publish_select_carrier(context, out_plan);
  }
  const bool result_is_mask = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, register_count);
  if (!result_is_mask) {
    return iree_ok_status();
  }

  loom_type_t condition_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op, condition,
                                                &condition_low_type));
  const bool condition_is_scc = loom_amdgpu_low_type_is_register_class_count(
      context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);
  const bool condition_is_sgpr_bool =
      loom_amdgpu_low_type_is_register_class_count(
          context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  const bool condition_is_mask = loom_amdgpu_low_type_is_register_class_count(
      context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (!condition_is_scc && !condition_is_sgpr_bool && !condition_is_mask) {
    return iree_ok_status();
  }

  loom_amdgpu_vector_select_plan_t plan = {
      .payload_kind = LOOM_AMDGPU_SELECT_PAYLOAD_KIND_I1_MASK,
      .condition_kind =
          condition_is_scc
              ? LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC
              : (condition_is_sgpr_bool
                     ? LOOM_AMDGPU_SELECT_CONDITION_KIND_SGPR_BOOL
                     : LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK),
      .condition = condition,
      .true_value = true_value,
      .false_value = false_value,
      .result = result,
      .lane_count = register_count,
      .registers_per_condition_lane = 1,
  };
  plan.payload.mask.operands.condition =
      loom_amdgpu_select_mask_operand_kind(context, condition);
  plan.payload.mask.operands.true_value =
      loom_amdgpu_select_mask_operand_kind(context, true_value);
  plan.payload.mask.operands.false_value =
      loom_amdgpu_select_mask_operand_kind(context, false_value);
  plan.payload.mask.recipe = loom_amdgpu_select_i1_mask_recipe(&plan);
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_i1_mask_select_descriptors(
      context, &plan, &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }
  *out_plan = plan;
  *out_selected = true;
  return loom_amdgpu_publish_select_carrier(context, out_plan);
}

iree_status_t loom_amdgpu_select_scf_select_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_select_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_scf_select_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (loom_amdgpu_type_is_i1(result_type) ||
      (loom_type_is_vector(result_type) &&
       loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1)) {
    return loom_amdgpu_select_scf_select_i1_plan(context, source_op, out_plan,
                                                 out_selected);
  }
  uint32_t register_count = 0;
  bool allows_lane_immediates = false;
  if (loom_type_is_buffer(result_type)) {
    register_count = 2;
  } else if (!loom_amdgpu_select_payload_storage(result_type, &register_count,
                                                 &allows_lane_immediates)) {
    return iree_ok_status();
  }
  const loom_value_id_t condition = loom_scf_select_condition(source_op);
  const loom_value_id_t true_value = loom_scf_select_true_value(source_op);
  const loom_value_id_t false_value = loom_scf_select_false_value(source_op);
  if (!loom_amdgpu_type_is_i1(loom_module_value_type(module, condition)) ||
      !loom_type_equal(loom_module_value_type(module, true_value),
                       result_type) ||
      !loom_type_equal(loom_module_value_type(module, false_value),
                       result_type)) {
    return iree_ok_status();
  }

  loom_type_t condition_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op, condition,
                                                &condition_low_type));
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, result, &result_low_type));
  if (!loom_low_type_is_register(result_low_type)) {
    return iree_ok_status();
  }
  if (loom_amdgpu_type_is_address_scalar(result_type)) {
    register_count = loom_low_register_type_unit_count(result_low_type);
    allows_lane_immediates = register_count == 1;
  }

  const bool condition_is_scc = loom_amdgpu_low_type_is_register_class_count(
      context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);
  const bool condition_is_sgpr_bool =
      loom_amdgpu_low_type_is_register_class_count(
          context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  const bool result_is_sgpr = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, register_count);
  if ((condition_is_scc || condition_is_sgpr_bool) && result_is_sgpr) {
    loom_low_lower_resolved_descriptor_t scc_descriptor = {0};
    bool scc_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32, &scc_descriptor,
        &scc_descriptor_present));
    if (!scc_descriptor_present) {
      return iree_ok_status();
    }
    loom_low_lower_resolved_descriptor_t sgpr_bool_compare_descriptor = {0};
    if (condition_is_sgpr_bool) {
      bool compare_present = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
          context, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32_SRC1_INLINE,
          &sgpr_bool_compare_descriptor, &compare_present));
      if (!compare_present) {
        return iree_ok_status();
      }
    }
    *out_plan = (loom_amdgpu_vector_select_plan_t){
        .payload_kind = LOOM_AMDGPU_SELECT_PAYLOAD_KIND_DATA,
        .condition_kind = condition_is_scc
                              ? LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC
                              : LOOM_AMDGPU_SELECT_CONDITION_KIND_SGPR_BOOL,
        .scc_descriptor = scc_descriptor,
        .sgpr_bool_compare_descriptor = sgpr_bool_compare_descriptor,
        .condition = condition,
        .true_value = true_value,
        .false_value = false_value,
        .result = result,
        .lane_count = register_count,
        .registers_per_condition_lane = 1,
    };
    *out_selected = true;
    return loom_amdgpu_publish_select_carrier(context, out_plan);
  }

  const bool condition_is_mask = loom_amdgpu_low_type_is_register_class_count(
      context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (!condition_is_mask) {
    return iree_ok_status();
  }

  loom_amdgpu_cndmask_b32_descriptors_t cndmask_descriptors = {0};
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_cndmask_b32_descriptors(
      context, LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_REGISTER,
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_ALL, &cndmask_descriptors,
      &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_vector_select_plan_t){
      .payload_kind = LOOM_AMDGPU_SELECT_PAYLOAD_KIND_DATA,
      .condition_kind = LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK,
      .cndmask_descriptors = cndmask_descriptors,
      .condition = condition,
      .true_value = true_value,
      .false_value = false_value,
      .result = result,
      .lane_count = register_count,
      .registers_per_condition_lane = 1,
  };
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_select_data_lanes(context, allows_lane_immediates, out_plan));
  *out_selected = true;
  return loom_amdgpu_publish_select_carrier(context, out_plan);
}

static iree_status_t loom_amdgpu_slice_lane_if_needed(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t lane_count, uint32_t unit_offset,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (lane_count == 1) {
    *out_lane = low_source;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source, unit_offset,
                                    lane_type, out_lane);
}

static iree_status_t loom_amdgpu_slice_source_lane_if_needed(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t lane_count, uint32_t unit_offset,
    loom_type_t fallback_lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (lane_count == 1) {
    *out_lane = low_source;
    return iree_ok_status();
  }
  loom_type_t source_lane_type = fallback_lane_type;
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), low_source);
  if (loom_type_is_register(source_type) &&
      loom_low_register_type_unit_count(source_type) > 1) {
    source_lane_type =
        loom_low_register_carrier_type_with_unit_count(source_type, 1);
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source, unit_offset,
                                    source_lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_select_lane_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    const loom_named_attr_t* attrs, iree_host_size_t attr_count,
    loom_type_t lane_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_op_t* lane_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &lane_op));
  *out_result = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_immediate_materialize_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_select_immediate_operand_kind_t kind,
    const loom_amdgpu_vector_select_plan_t* plan,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t condition, uint32_t lane, loom_type_t lane_type,
    loom_value_id_t* out_operand) {
  *out_operand = LOOM_VALUE_ID_INVALID;
  switch (kind) {
    case LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_CONDITION:
      *out_operand = condition;
      return iree_ok_status();
    case LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_FALSE_LANE: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
          context, source_op, low_false_value, plan->lane_count, lane,
          lane_type, out_operand));
      return loom_amdgpu_materialize_full_low_vgpr_b32(
          context, source_op, *out_operand, out_operand);
    }
    case LOOM_AMDGPU_SELECT_IMMEDIATE_OPERAND_TRUE_LANE: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
          context, source_op, low_true_value, plan->lane_count, lane, lane_type,
          out_operand));
      return loom_amdgpu_materialize_full_low_vgpr_b32(
          context, source_op, *out_operand, out_operand);
    }
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU select immediate operand kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_emit_vector_select_immediate_candidate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan,
    const loom_amdgpu_select_immediate_candidate_t* candidate,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t condition, uint32_t lane, loom_type_t lane_type,
    uint32_t false_bits, uint32_t true_bits, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[2] = {0};
  for (uint8_t i = 0; i < candidate->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_immediate_materialize_operand(
        context, source_op, candidate->operands[i], plan, low_false_value,
        low_true_value, condition, lane, lane_type, &operands[i]));
  }

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  for (uint8_t i = 0; i < candidate->attr_count; ++i) {
    const loom_amdgpu_select_immediate_attr_t* attr = &candidate->attrs[i];
    const uint32_t bits =
        attr->value_role == LOOM_AMDGPU_SELECT_IMMEDIATE_VALUE_FALSE
            ? false_bits
            : true_bits;
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, loom_amdgpu_select_immediate_attr_name(attr->kind), bits,
        attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  }

  return loom_amdgpu_emit_select_lane_op(
      context, source_op,
      loom_amdgpu_select_immediate_candidate_descriptor(plan, candidate),
      operands, candidate->operand_count, attrs, attr_count, lane_type,
      out_result);
}

static iree_status_t loom_amdgpu_emit_i1_mask_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &op));
  *out_result = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_i1_mask_exec_read(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan, loom_type_t mask_type,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_op_t* exec_read_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->payload.mask.exec_read_descriptor,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      &mask_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &exec_read_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(exec_read_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_sgpr_bool_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan, loom_value_id_t low_condition,
    loom_value_id_t* out_low_condition) {
  *out_low_condition = LOOM_VALUE_ID_INVALID;
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("rhs"), 0, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->sgpr_bool_compare_descriptor, &low_condition, 1,
      loom_make_named_attr_slice(attrs, attr_count), &scc_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &compare_op));
  *out_low_condition = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_i1_mask_invert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan, loom_value_id_t low_mask,
    loom_type_t mask_type, loom_value_id_t* out_inverse_mask) {
  loom_value_id_t exec_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i1_mask_exec_read(
      context, source_op, plan, mask_type, &exec_mask));
  return loom_amdgpu_emit_i1_mask_binary(
      context, source_op, &plan->payload.mask.xor_descriptor, low_mask,
      exec_mask, mask_type, out_inverse_mask);
}

static iree_status_t loom_amdgpu_materialize_selected_mask_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan, loom_value_id_t source_value,
    uint8_t operand_kind, uint32_t mask_index, uint32_t mask_count,
    loom_type_t mask_type, loom_value_id_t* out_value) {
  if (operand_kind == LOOM_AMDGPU_MASK_OPERAND_ZERO) {
    return loom_amdgpu_emit_sgpr64_constant_u64(context, source_op, 0,
                                                out_value);
  }
  if (operand_kind == LOOM_AMDGPU_MASK_OPERAND_EXEC) {
    return loom_amdgpu_emit_i1_mask_exec_read(context, source_op, plan,
                                              mask_type, out_value);
  }
  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);
  if (mask_count > 1) {
    return loom_amdgpu_emit_low_slice(context, source_op, low_value,
                                      mask_index * 2u, mask_type, out_value);
  }
  return loom_amdgpu_materialize_low_native_i1_mask(context, source_op,
                                                    low_value, out_value);
}

static bool loom_amdgpu_mask_select_recipe_needs_inverse_condition(
    loom_amdgpu_mask_select_recipe_t recipe) {
  switch (recipe) {
    case LOOM_AMDGPU_MASK_SELECT_INVERSE_CONDITION:
    case LOOM_AMDGPU_MASK_SELECT_INVERSE_AND_FALSE:
    case LOOM_AMDGPU_MASK_SELECT_TRUE_OR_INVERSE:
    case LOOM_AMDGPU_MASK_SELECT_MERGE:
      return true;
    case LOOM_AMDGPU_MASK_SELECT_TRUE_VALUE:
    case LOOM_AMDGPU_MASK_SELECT_UNIFORM:
    case LOOM_AMDGPU_MASK_SELECT_CONDITION:
    case LOOM_AMDGPU_MASK_SELECT_CONDITION_OR_FALSE:
    case LOOM_AMDGPU_MASK_SELECT_CONDITION_AND_TRUE:
      return false;
  }
  return false;
}

static iree_status_t loom_amdgpu_lower_i1_mask_select_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan, loom_value_id_t low_condition,
    loom_value_id_t low_inverse_condition, uint32_t mask_index,
    uint32_t mask_count, loom_type_t lane_type, loom_type_t mask_type,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_amdgpu_mask_select_recipe_t recipe =
      (loom_amdgpu_mask_select_recipe_t)plan->payload.mask.recipe;
  if (plan->true_value == plan->false_value) {
    return loom_amdgpu_materialize_selected_mask_operand(
        context, source_op, plan, plan->true_value,
        plan->payload.mask.operands.true_value, mask_index, mask_count,
        mask_type, out_result);
  }

  loom_value_id_t low_true_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_false_value = LOOM_VALUE_ID_INVALID;
  if (recipe == LOOM_AMDGPU_MASK_SELECT_UNIFORM) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_selected_mask_operand(
        context, source_op, plan, plan->true_value,
        plan->payload.mask.operands.true_value, mask_index, mask_count,
        mask_type, &low_true_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_selected_mask_operand(
        context, source_op, plan, plan->false_value,
        plan->payload.mask.operands.false_value, mask_index, mask_count,
        mask_type, &low_false_value));
    loom_value_id_t lane_results[2] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
    };
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(lane_results); ++i) {
      loom_value_id_t lane_true_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_true_value, IREE_ARRAYSIZE(lane_results), i,
          lane_type, &lane_true_value));
      loom_value_id_t lane_false_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_false_value, IREE_ARRAYSIZE(lane_results), i,
          lane_type, &lane_false_value));
      const loom_value_id_t operands[] = {
          lane_true_value,
          lane_false_value,
          low_condition,
      };
      loom_op_t* select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->scc_descriptor, operands, IREE_ARRAYSIZE(operands),
          loom_named_attr_slice_empty(), &lane_type, 1,
          /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
          &select_op));
      lane_results[i] = loom_value_slice_get(loom_low_op_results(select_op), 0);
    }
    return loom_amdgpu_build_low_register_range(
        context, source_op, lane_results, IREE_ARRAYSIZE(lane_results),
        mask_type, out_result);
  }

  IREE_ASSERT_EQ(plan->condition_kind,
                 LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK);
  if (recipe == LOOM_AMDGPU_MASK_SELECT_TRUE_VALUE) {
    return loom_amdgpu_materialize_selected_mask_operand(
        context, source_op, plan, plan->true_value,
        plan->payload.mask.operands.true_value, mask_index, mask_count,
        mask_type, out_result);
  }
  if (recipe == LOOM_AMDGPU_MASK_SELECT_CONDITION) {
    *out_result = low_condition;
    return iree_ok_status();
  }
  if (recipe == LOOM_AMDGPU_MASK_SELECT_INVERSE_CONDITION) {
    *out_result = low_inverse_condition;
    return iree_ok_status();
  }

  if (recipe == LOOM_AMDGPU_MASK_SELECT_CONDITION_OR_FALSE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_selected_mask_operand(
        context, source_op, plan, plan->false_value,
        plan->payload.mask.operands.false_value, mask_index, mask_count,
        mask_type, &low_false_value));
    return loom_amdgpu_emit_i1_mask_binary(
        context, source_op, &plan->payload.mask.or_descriptor, low_condition,
        low_false_value, mask_type, out_result);
  }

  if (recipe == LOOM_AMDGPU_MASK_SELECT_INVERSE_AND_FALSE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_selected_mask_operand(
        context, source_op, plan, plan->false_value,
        plan->payload.mask.operands.false_value, mask_index, mask_count,
        mask_type, &low_false_value));
    return loom_amdgpu_emit_i1_mask_binary(
        context, source_op, &plan->payload.mask.and_descriptor,
        low_inverse_condition, low_false_value, mask_type, out_result);
  }

  if (recipe == LOOM_AMDGPU_MASK_SELECT_TRUE_OR_INVERSE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_selected_mask_operand(
        context, source_op, plan, plan->true_value,
        plan->payload.mask.operands.true_value, mask_index, mask_count,
        mask_type, &low_true_value));
    return loom_amdgpu_emit_i1_mask_binary(
        context, source_op, &plan->payload.mask.or_descriptor, low_true_value,
        low_inverse_condition, mask_type, out_result);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_selected_mask_operand(
      context, source_op, plan, plan->true_value,
      plan->payload.mask.operands.true_value, mask_index, mask_count, mask_type,
      &low_true_value));
  loom_value_id_t true_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i1_mask_binary(
      context, source_op, &plan->payload.mask.and_descriptor, low_condition,
      low_true_value, mask_type, &true_mask));
  if (recipe == LOOM_AMDGPU_MASK_SELECT_CONDITION_AND_TRUE) {
    *out_result = true_mask;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_selected_mask_operand(
      context, source_op, plan, plan->false_value,
      plan->payload.mask.operands.false_value, mask_index, mask_count,
      mask_type, &low_false_value));
  loom_value_id_t false_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i1_mask_binary(
      context, source_op, &plan->payload.mask.and_descriptor,
      low_inverse_condition, low_false_value, mask_type, &false_mask));
  return loom_amdgpu_emit_i1_mask_binary(
      context, source_op, &plan->payload.mask.or_descriptor, true_mask,
      false_mask, mask_type, out_result);
}

static iree_status_t loom_amdgpu_lower_i1_mask_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan) {
  IREE_ASSERT_GT(plan->lane_count, 0u);
  IREE_ASSERT_EQ(plan->lane_count % 2u, 0u);
  const uint32_t mask_count = plan->lane_count / 2u;
  IREE_ASSERT_LE(mask_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &lane_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t low_condition =
      loom_low_lower_lookup_value(context, plan->condition);
  if (plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SGPR_BOOL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_bool_scc(
        context, source_op, plan, low_condition, &low_condition));
  } else if (plan->condition_kind ==
             LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_selected_mask_operand(
        context, source_op, plan, plan->condition,
        plan->payload.mask.operands.condition, /*mask_index=*/0,
        /*mask_count=*/1, mask_type, &low_condition));
  }

  const loom_amdgpu_mask_select_recipe_t recipe =
      (loom_amdgpu_mask_select_recipe_t)plan->payload.mask.recipe;
  loom_value_id_t low_inverse_condition = LOOM_VALUE_ID_INVALID;
  if (plan->true_value != plan->false_value &&
      loom_amdgpu_mask_select_recipe_needs_inverse_condition(recipe)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_i1_mask_invert(context, source_op, plan, low_condition,
                                        mask_type, &low_inverse_condition));
  }

  loom_value_id_t low_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  const bool result_is_constant_splat =
      plan->payload.mask.operands.true_value !=
          LOOM_AMDGPU_MASK_OPERAND_VALUE &&
      plan->payload.mask.operands.false_value != LOOM_AMDGPU_MASK_OPERAND_VALUE;
  const uint32_t emitted_mask_count = result_is_constant_splat ? 1 : mask_count;
  for (uint32_t i = 0; i < emitted_mask_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_i1_mask_select_lane(
        context, source_op, plan, low_condition, low_inverse_condition, i,
        mask_count, lane_type, mask_type, &low_results[i]));
  }
  for (uint32_t i = emitted_mask_count; i < mask_count; ++i) {
    low_results[i] = low_results[0];
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             low_results, mask_count);
}

// Address facts may narrow either input independently of the joined result.
// Bring both carriers to the result width before selecting individual words.
static iree_status_t loom_amdgpu_resize_select_address_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, uint32_t register_count,
    loom_value_id_t* out_low_value) {
  *out_low_value = low_value;
  const loom_type_t type =
      loom_module_value_type(loom_low_lower_context_module(context), low_value);
  if (loom_low_register_type_unit_count(type) == register_count) {
    return iree_ok_status();
  }
  if (register_count == 1) {
    return loom_amdgpu_emit_low_slice(
        context, source_op, low_value, 0,
        loom_low_register_carrier_type_with_unit_count(type, 1), out_low_value);
  }
  if (loom_low_register_type_class_id(type) == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return loom_amdgpu_emit_sgpr64_from_u32(context, source_op, low_value,
                                            out_low_value);
  }
  return loom_amdgpu_emit_vgpr64_from_u32(context, source_op, low_value,
                                          out_low_value);
}

// Each predicate chooses the original packed word. Only that predicate's
// element bits enter the result, so payload extraction and repacking are
// unnecessary. The first element seeds the word; unused tail bits are padding.
static iree_status_t loom_amdgpu_lower_packed_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan) {
  loom_value_id_t condition =
      loom_low_lower_lookup_value(context, plan->condition);
  loom_value_id_t true_value =
      loom_low_lower_lookup_value(context, plan->true_value);
  loom_value_id_t false_value =
      loom_low_lower_lookup_value(context, plan->false_value);
  loom_type_t word_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &word_type));
  loom_type_t condition_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &condition_type));
  loom_type_t mask_type = loom_type_none();
  if (plan->payload.packed.mask_constant_descriptor.descriptor) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &mask_type));
  }

  const uint32_t element_bit_count = plan->payload.packed.element_bit_count;
  const uint32_t elements_per_word = 32u / element_bit_count;
  const uint32_t element_mask = (1u << element_bit_count) - 1u;
  loom_value_id_t words[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t word = 0; word < plan->lane_count; ++word) {
    loom_value_id_t true_word = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
        context, source_op, true_value, plan->lane_count, word, word_type,
        &true_word));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
        context, source_op, true_word, &true_word));
    loom_value_id_t false_word = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
        context, source_op, false_value, plan->lane_count, word, word_type,
        &false_word));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
        context, source_op, false_word, &false_word));

    const uint32_t first_element = word * elements_per_word;
    const uint32_t element_count = iree_min(
        elements_per_word, plan->payload.packed.element_count - first_element);
    loom_value_id_t merged = LOOM_VALUE_ID_INVALID;
    for (uint32_t element = 0; element < element_count; ++element) {
      loom_value_id_t element_condition = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, condition, plan->payload.packed.element_count,
          (first_element + element) * 2u, condition_type, &element_condition));
      const loom_value_id_t operands[] = {
          false_word,
          true_word,
          element_condition,
      };
      loom_value_id_t selected = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_select_lane_op(
          context, source_op, &plan->cndmask_descriptors.register_descriptor,
          operands, IREE_ARRAYSIZE(operands), NULL, 0, word_type, &selected));
      if (element == 0) {
        merged = selected;
        continue;
      }
      const uint32_t mask_bits = element_mask << (element * element_bit_count);
      if (plan->payload.packed.mask_constant_descriptor.descriptor) {
        loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
            context, source_op, &plan->payload.packed.mask_constant_descriptor,
            plan->payload.packed.imm32_attr_name_id, mask_bits, mask_type,
            &mask));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
            context, source_op, &plan->payload.packed.merge_descriptor, mask,
            selected, merged, word_type, &merged));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary_immediate(
            context, source_op, &plan->payload.packed.merge_descriptor,
            selected, merged, mask_bits, word_type, &merged));
      }
    }
    words[word] = merged;
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             words, plan->lane_count);
}

static iree_status_t loom_amdgpu_lower_uniform_i1_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan) {
  loom_value_id_t operands[3];
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_i1_integer(
      context, source_op, plan->true_value, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      &operands[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_i1_integer(
      context, source_op, plan->false_value, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      &operands[1]));
  operands[2] = loom_low_lower_lookup_value(context, plan->condition);
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_uniform_i1_condition(
      context, source_op, operands[2], &operands[2]));
  loom_type_t word_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &word_type));
  loom_value_id_t selected = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_select_lane_op(
      context, source_op, &plan->scc_descriptor, operands,
      IREE_ARRAYSIZE(operands), NULL, 0, word_type, &selected));
  return loom_low_lower_bind_value(context, plan->result, selected);
}

iree_status_t loom_amdgpu_lower_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan) {
  if (plan->payload_kind == LOOM_AMDGPU_SELECT_PAYLOAD_KIND_I1_BOOL) {
    return loom_amdgpu_lower_uniform_i1_select(context, source_op, plan);
  }
  if (plan->payload_kind == LOOM_AMDGPU_SELECT_PAYLOAD_KIND_I1_MASK) {
    return loom_amdgpu_lower_i1_mask_select(context, source_op, plan);
  }
  if (plan->payload_kind == LOOM_AMDGPU_SELECT_PAYLOAD_KIND_PACKED_DATA) {
    return loom_amdgpu_lower_packed_select(context, source_op, plan);
  }

  const uint32_t lane_count = plan->lane_count;
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_value_id_t low_condition =
      loom_low_lower_lookup_value(context, plan->condition);
  if (plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SGPR_BOOL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_bool_scc(
        context, source_op, plan, low_condition, &low_condition));
  }
  loom_value_id_t low_true_value =
      loom_low_lower_lookup_value(context, plan->true_value);
  loom_value_id_t low_false_value =
      loom_low_lower_lookup_value(context, plan->false_value);

  if (loom_amdgpu_value_is_address_scalar(context, plan->result)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resize_select_address_operand(
        context, source_op, low_true_value, lane_count, &low_true_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_resize_select_address_operand(
        context, source_op, low_false_value, lane_count, &low_false_value));
  }

  if (plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC ||
      plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SGPR_BOOL) {
    loom_type_t lane_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &lane_type));
    loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    for (uint32_t i = 0; i < lane_count; ++i) {
      loom_value_id_t lane_true_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_true_value, lane_count, i, lane_type,
          &lane_true_value));
      loom_value_id_t lane_false_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_false_value, lane_count, i, lane_type,
          &lane_false_value));
      if (lane_true_value == lane_false_value) {
        lane_results[i] = lane_true_value;
        continue;
      }
      const loom_value_id_t operands[] = {
          lane_true_value,
          lane_false_value,
          low_condition,
      };
      loom_op_t* select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &plan->scc_descriptor, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
          /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
          &select_op));
      lane_results[i] = loom_value_slice_get(loom_low_op_results(select_op), 0);
    }
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               lane_results, lane_count);
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  uint8_t selected_lane_index = 0;
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane_condition = LOOM_VALUE_ID_INVALID;
    if (plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK) {
      lane_condition = low_condition;
    } else {
      const uint32_t condition_lane = i / plan->registers_per_condition_lane;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_condition, lane_count, condition_lane * 2u,
          mask_lane_type, &lane_condition));
    }

    if (selected_lane_index < plan->payload.data.lane_count &&
        plan->payload.data.lanes[selected_lane_index].lane == i) {
      const loom_amdgpu_select_lane_plan_t* selected =
          &plan->payload.data.lanes[selected_lane_index++];
      if (selected->recipe == 0) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
            context, source_op, low_true_value, lane_count, i, lane_type,
            &lane_results[i]));
        IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
            context, source_op, lane_results[i], &lane_results[i]));
      } else {
        const loom_amdgpu_select_immediate_candidate_t* candidate =
            &kLoomAmdgpuSelectImmediateCandidates[selected->recipe - 1u];
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_select_immediate_candidate(
            context, source_op, plan, candidate, low_false_value,
            low_true_value, lane_condition, i, lane_type, selected->bits[0],
            selected->bits[1], &lane_results[i]));
      }
      continue;
    }

    loom_value_id_t lane_true_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
        context, source_op, low_true_value, lane_count, i, lane_type,
        &lane_true_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
        context, source_op, lane_true_value, &lane_true_value));
    loom_value_id_t lane_false_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_source_lane_if_needed(
        context, source_op, low_false_value, lane_count, i, lane_type,
        &lane_false_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
        context, source_op, lane_false_value, &lane_false_value));
    if (lane_false_value == lane_true_value) {
      lane_results[i] = lane_true_value;
      continue;
    }
    const loom_value_id_t operands[] = {
        lane_false_value,
        lane_true_value,
        lane_condition,
    };
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->cndmask_descriptors.register_descriptor, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
        &lane_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &lane_op));
    lane_results[i] = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, lane_count, &result_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, source_op, lane_results, lane_count, result_type, &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}
