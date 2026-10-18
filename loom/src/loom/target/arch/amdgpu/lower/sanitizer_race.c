// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race.h"

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/atomic.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/address.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/emit.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/packet.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/memory_packet.h"
#include "loom/target/arch/amdgpu/lower/sanitizer.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_history_policy.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_lifetime.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_report.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_state.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_report.h"
#include "loom/target/arch/amdgpu/lower/source_alloca_layout.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/sync.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

typedef struct loom_amdgpu_sanitizer_race_access_range_t {
  // Memory-space-relative base byte offset of the physical access.
  loom_value_id_t memory_byte_offset;
  // Nonzero contiguous physical access length in bytes.
  uint32_t byte_length;
} loom_amdgpu_sanitizer_race_access_range_t;

typedef struct loom_amdgpu_sanitizer_race_fragment_access_t {
  // Exact target-selected physical fragment access.
  loom_amdgpu_fragment_memory_issued_access_t access;
  // Contiguous byte length covered by the physical access.
  uint32_t byte_length;
} loom_amdgpu_sanitizer_race_fragment_access_t;

enum loom_amdgpu_sanitizer_race_fragment_access_capacity_e {
  // Maximum physical accesses represented by one selected fragment plan.
  LOOM_AMDGPU_SANITIZER_RACE_FRAGMENT_ACCESS_CAPACITY =
      LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS *
      LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET,
};

typedef struct loom_amdgpu_sanitizer_race_shadow_address_t {
  // Per-workgroup shadow header byte offset from shadow_base.
  loom_value_id_t workgroup_offset;
  // Per-access shadow-entry byte offset from shadow_base.
  loom_value_id_t entry_offset;
} loom_amdgpu_sanitizer_race_shadow_address_t;

typedef struct loom_amdgpu_sanitizer_race_shadow_entry_t {
  // Low 32 bits of the compact TSAN shadow entry.
  loom_value_id_t low;
  // High 32 bits of the compact TSAN shadow entry.
  loom_value_id_t high;
  // Combined 64-bit TSAN shadow entry.
  loom_value_id_t value;
} loom_amdgpu_sanitizer_race_shadow_entry_t;

typedef struct loom_amdgpu_sanitizer_race_history_relation_t {
  // Decoded access kind carried by the history.
  loom_value_id_t access_kind;
  // Active lanes whose history belongs to the current epoch and dispatch.
  loom_value_id_t valid_mask;
  // Active lanes whose history has the same workitem and access category.
  loom_value_id_t same_workitem_category_mask;
  // Active lanes whose history is a write-like access by this workitem.
  loom_value_id_t same_workitem_write_mask;
  // Active lanes whose history conflicts with the current access.
  loom_value_id_t failure_mask;
} loom_amdgpu_sanitizer_race_history_relation_t;

typedef struct loom_amdgpu_sanitizer_race_publication_t {
  // Active lanes that must publish the current history.
  loom_value_id_t publish_mask;
  // Publishing lanes that selected the second history slot.
  loom_value_id_t second_slot_mask;
  // History value expected in the selected slot.
  loom_amdgpu_sanitizer_race_shadow_entry_t expected;
} loom_amdgpu_sanitizer_race_publication_t;

static uint32_t loom_amdgpu_sanitizer_race_shadow_bit_mask(uint32_t bit_shift,
                                                           uint32_t bit_count) {
  if (bit_count == 32) {
    return UINT32_MAX;
  }
  return (((uint32_t)1u << bit_count) - 1u) << bit_shift;
}

static uint32_t loom_amdgpu_sanitizer_race_shadow_low_epoch_generation_mask(
    void) {
  const uint32_t bit_count = 32u - LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT;
  return loom_amdgpu_sanitizer_race_shadow_bit_mask(
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT, bit_count);
}

static uint32_t loom_amdgpu_sanitizer_race_shadow_high_generation_mask(void) {
  return loom_amdgpu_sanitizer_race_shadow_bit_mask(
      /*bit_shift=*/0, LOOM_AMDGPU_TSAN_SHADOW_ENTRY_HIGH_GENERATION_BIT_COUNT);
}

static uint32_t loom_amdgpu_sanitizer_race_shadow_workitem_mask(void) {
  return loom_amdgpu_sanitizer_race_shadow_bit_mask(
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_BIT_COUNT);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_descriptor_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_location_id_t location,
    loom_op_t** out_op) {
  *out_op = NULL;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0, operands,
      operand_count, attrs, result_types, result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, out_op);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_global_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t register_count,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;

  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, register_count,
      &result_type));
  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_offset_attr(
      builder, 0, &attrs[attr_count++]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_load_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/1,
                                               attrs, &attr_count);

  const loom_value_id_t operands[] = {byte_offset, base_address};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  const loom_value_id_t value =
      loom_value_slice_get(loom_low_op_results(op), 0);
  if (iree_any_bit_set(flags, LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_ACQUIRE)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_system_memory_build_acquire_ordering_scoped(
            builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location));
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_global_load_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_amdgpu_sanitizer_race_build_global_load(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR,
      /*register_count=*/1, base_address, byte_offset, flags, location,
      out_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_global_load_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_amdgpu_sanitizer_race_build_global_load(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
      /*register_count=*/2, base_address, byte_offset, flags, location,
      out_value);
}

static iree_status_t
loom_amdgpu_sanitizer_race_build_global_compare_exchange_u64_acq_rel(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_value_id_t expected_value, loom_value_id_t desired_value,
    loom_location_id_t location, loom_value_id_t* out_observed_value) {
  *out_observed_value = LOOM_VALUE_ID_INVALID;

  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR);

  loom_value_id_t expected_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, expected_value, /*expected_unit_count=*/2,
      location, &expected_vgpr));
  loom_value_id_t desired_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, desired_value, /*expected_unit_count=*/2,
      location, &desired_vgpr));
  loom_type_t compare_exchange_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 4,
      &compare_exchange_type));
  // AMDGPU compare-and-swap consumes the desired value in the low lanes and
  // the expected value in the high lanes.
  const loom_value_id_t compare_exchange_parts[] = {desired_vgpr,
                                                    expected_vgpr};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      builder, compare_exchange_parts, IREE_ARRAYSIZE(compare_exchange_parts),
      compare_exchange_type, location, &concat_op));
  const loom_value_id_t compare_exchange_value =
      loom_low_concat_result(concat_op);

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_atomic_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));

  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location));

  const loom_value_id_t operands[] = {byte_offset, compare_exchange_value,
                                      base_address};

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &result_type));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  const loom_value_id_t observed =
      loom_value_slice_get(loom_low_op_results(op), 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_acquire_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location));
  *out_observed_value = observed;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_mask);

static iree_status_t loom_amdgpu_sanitizer_race_mask_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value);

static iree_status_t loom_amdgpu_sanitizer_race_vgpr_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_low_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
                                    vgpr_type, out_low_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_concat_vgpr_u64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low, loom_value_id_t high, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  const loom_value_id_t parts[] = {low, high};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), parts, IREE_ARRAYSIZE(parts),
      vgpr_x2_type, source_op->location, &concat_op));
  *out_low_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_mask_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t condition_mask, loom_value_id_t* out_value) {
  loom_value_id_t difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64, false_value,
      true_value, &difference));
  loom_value_id_t selected_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, difference,
      condition_mask, &selected_difference));
  return loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64, false_value,
      selected_difference, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_vgpr_u32_or_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t immediate, loom_value_id_t* out_low_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  if (immediate == 0) {
    *out_low_value = value;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT, value,
      immediate, vgpr_type, out_low_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_vgpr_u32_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t condition_mask, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  const loom_value_id_t operands[] = {false_value, true_value, condition_mask};
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &vgpr_type, 1,
      &select_op));
  *out_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_shadow_entry_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* false_entry,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* true_entry,
    loom_value_id_t condition_mask,
    loom_amdgpu_sanitizer_race_shadow_entry_t* out_entry) {
  *out_entry = (loom_amdgpu_sanitizer_race_shadow_entry_t){0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_select(
      context, source_op, false_entry->low, true_entry->low, condition_mask,
      &out_entry->low));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_select(
      context, source_op, false_entry->high, true_entry->high, condition_mask,
      &out_entry->high));
  return loom_amdgpu_sanitizer_race_concat_vgpr_u64(
      context, source_op, out_entry->low, out_entry->high, &out_entry->value);
}

static iree_status_t loom_amdgpu_sanitizer_race_mask_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &op));
  *out_mask = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_exec_narrow_and_save(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lane_mask, loom_location_id_t location,
    loom_value_id_t* out_saved_exec) {
  *out_saved_exec = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &scc_type));
  const loom_type_t result_types[] = {sgpr_x2_type, scc_type};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &lane_mask, /*operand_count=*/1, loom_named_attr_slice_empty(),
      result_types, IREE_ARRAYSIZE(result_types), location, &op));
  *out_saved_exec = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_exec_read(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location, loom_value_id_t* out_exec) {
  *out_exec = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      &sgpr_x2_type, /*result_count=*/1, location, &op));
  *out_exec = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_exec_restore(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t saved_exec, loom_location_id_t location) {
  loom_op_t* op = NULL;
  return loom_amdgpu_sanitizer_race_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, /*operand_count=*/1, loom_named_attr_slice_empty(),
      /*result_types=*/NULL, /*result_count=*/0, location, &op);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_access_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access,
    loom_value_id_t* out_byte_offset) {
  loom_amdgpu_memory_dynamic_term_sequence_t sequence = {0};
  loom_amdgpu_memory_access_resolve_dynamic_terms(context, access, &sequence);
  return loom_amdgpu_emit_memory_vaddr(context, source_op, access, &sequence,
                                       LOOM_VALUE_ID_INVALID, out_byte_offset);
}

static iree_status_t
loom_amdgpu_sanitizer_race_build_shadow_entry_address_from_index(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t workgroup_shadow_record_offset,
    loom_value_id_t workgroup_shadow_history_offset,
    loom_value_id_t entry_index,
    loom_amdgpu_sanitizer_race_shadow_address_t* out_address) {
  *out_address = (loom_amdgpu_sanitizer_race_shadow_address_t){0};
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_amdgpu_sanitizer_race_shadow_address_t address = {0};
  address.workgroup_offset = workgroup_shadow_record_offset;
  loom_value_id_t entry_data_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 4,
      entry_index, vgpr_type, &entry_data_offset));
  loom_value_id_t entry_data_offset_wide = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_from_u32(
      context, source_op, entry_data_offset, &entry_data_offset_wide));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
      context, source_op, workgroup_shadow_history_offset,
      entry_data_offset_wide, &address.entry_offset));
  *out_address = address;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_shadow_report_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_shadow_address_t* shadow_address,
    loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  loom_value_id_t shadow_base_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      loom_low_lower_context_builder(context),
      loom_low_lower_context_descriptor_set(context), config->shadow_base,
      /*expected_unit_count=*/2, source_op->location, &shadow_base_vgpr));
  return loom_amdgpu_emit_vgpr64_add(context, source_op, shadow_base_vgpr,
                                     shadow_address->entry_offset, out_address);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_shadow_entry(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_tsan_shadow_access_kind_t access_kind,
    const loom_amdgpu_sanitizer_race_shadow_entry_base_t* base,
    loom_value_id_t epoch, loom_sanitizer_site_id_t site_id,
    loom_amdgpu_sanitizer_race_shadow_entry_t* out_entry) {
  *out_entry = (loom_amdgpu_sanitizer_race_shadow_entry_t){0};
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t epoch_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, epoch,
      LOOM_AMDGPU_TSAN_SHADOW_EPOCH_CAPACITY - 1u, vgpr_type, &epoch_low));
  loom_value_id_t epoch_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT, epoch_low, vgpr_type,
      &epoch_bits));

  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, access_kind, &low));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low, base->low,
      vgpr_type, &low));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low, epoch_bits,
      vgpr_type, &low));

  loom_value_id_t high = base->high;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_or_immediate(
      context, source_op, high,
      (site_id & ((1u << LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_BIT_COUNT) - 1u))
          << (LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_SHIFT - 32u),
      &high));

  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_u64(
      context, source_op, low, high, &value));
  *out_entry = (loom_amdgpu_sanitizer_race_shadow_entry_t){
      .low = low,
      .high = high,
      .value = value,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_unpack_shadow_entry(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t entry_value,
    loom_amdgpu_sanitizer_race_shadow_entry_t* out_entry) {
  *out_entry = (loom_amdgpu_sanitizer_race_shadow_entry_t){0};

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, entry_value, /*lane_offset=*/0, vgpr_type, &low));
  loom_value_id_t high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, entry_value, /*lane_offset=*/1, vgpr_type, &high));
  *out_entry = (loom_amdgpu_sanitizer_race_shadow_entry_t){
      .low = low,
      .high = high,
      .value = entry_value,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_decode_shadow_access_kind(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* entry,
    loom_value_id_t* out_access_kind) {
  *out_access_kind = LOOM_VALUE_ID_INVALID;

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, entry->low,
      (1u << LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT) - 1u,
      vgpr_type, out_access_kind);
}

static iree_status_t loom_amdgpu_sanitizer_race_decode_shadow_workitem(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t entry_value, loom_value_id_t* out_workitem) {
  *out_workitem = LOOM_VALUE_ID_INVALID;

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, entry_value, /*lane_offset=*/0, vgpr_type, &low));
  loom_value_id_t shifted_workitem = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT, low, vgpr_type,
      &shifted_workitem));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      shifted_workitem,
      (1u << LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_BIT_COUNT) - 1u, vgpr_type,
      out_workitem));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_decode_shadow_site_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t entry_value, loom_value_id_t* out_site_id) {
  *out_site_id = LOOM_VALUE_ID_INVALID;

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, entry_value, /*lane_offset=*/1, vgpr_type, &high));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_SHIFT - 32u, high, vgpr_type,
      out_site_id);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_current_epoch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    loom_value_id_t workgroup_shadow_offset, loom_value_id_t* out_epoch) {
  *out_epoch = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t epoch_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_low_slice(context, source_op, workgroup_shadow_offset,
                                 /*lane_offset=*/0, vgpr_type, &epoch_offset));
  return loom_amdgpu_sanitizer_race_build_global_load_b32(
      loom_low_lower_context_builder(context),
      loom_low_lower_context_descriptor_set(context), config->shadow_base,
      epoch_offset, LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_ACQUIRE,
      source_op->location, out_epoch);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_shadow_entry_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_shadow_address_t* shadow_address,
    loom_value_id_t* out_offset) {
  *out_offset = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_low_slice(context, source_op,
                                    shadow_address->entry_offset,
                                    /*lane_offset=*/0, vgpr_type, out_offset);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_history_relation(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_observation_plan_t* plan,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* current_entry,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* observed_entry,
    loom_value_id_t active_mask,
    loom_amdgpu_sanitizer_race_history_relation_t* out_relation) {
  *out_relation = (loom_amdgpu_sanitizer_race_history_relation_t){0};

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t prior_access_kind = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_decode_shadow_access_kind(
      context, source_op, observed_entry, &prior_access_kind));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, 0, &zero));

  loom_value_id_t prior_nonempty_raw = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      prior_access_kind, zero, &prior_nonempty_raw));
  loom_value_id_t prior_nonempty = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      prior_nonempty_raw, active_mask, &prior_nonempty));

  loom_value_id_t low_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32,
      observed_entry->low, current_entry->low, vgpr_type, &low_difference));

  loom_value_id_t epoch_generation_low_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      low_difference,
      loom_amdgpu_sanitizer_race_shadow_low_epoch_generation_mask(), vgpr_type,
      &epoch_generation_low_difference));
  loom_value_id_t high_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32,
      observed_entry->high, current_entry->high, vgpr_type, &high_difference));
  loom_value_id_t epoch_generation_high_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      high_difference, loom_amdgpu_sanitizer_race_shadow_high_generation_mask(),
      vgpr_type, &epoch_generation_high_difference));
  loom_value_id_t epoch_generation_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
      epoch_generation_low_difference, epoch_generation_high_difference,
      vgpr_type, &epoch_generation_difference));
  loom_value_id_t same_epoch_generation_raw = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      epoch_generation_difference, zero, &same_epoch_generation_raw));
  loom_value_id_t same_epoch_generation = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      same_epoch_generation_raw, active_mask, &same_epoch_generation));
  loom_value_id_t valid_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, prior_nonempty,
      same_epoch_generation, &valid_mask));

  loom_value_id_t workitem_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      low_difference, loom_amdgpu_sanitizer_race_shadow_workitem_mask(),
      vgpr_type, &workitem_difference));
  loom_value_id_t same_workitem_raw = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      workitem_difference, zero, &same_workitem_raw));
  loom_value_id_t same_workitem = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, valid_mask,
      same_workitem_raw, &same_workitem));

  loom_value_id_t read_kind = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ,
      &read_kind));
  loom_value_id_t atomic_kind = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC,
      &atomic_kind));
  loom_value_id_t write_kind = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE,
      &write_kind));
  loom_value_id_t read_write_kind = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ_WRITE,
      &read_write_kind));

  loom_value_id_t prior_write = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      prior_access_kind, write_kind, &prior_write));
  loom_value_id_t prior_read_write = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      prior_access_kind, read_write_kind, &prior_read_write));
  loom_value_id_t prior_write_like = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64, prior_write,
      prior_read_write, &prior_write_like));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      prior_write_like, same_workitem,
      &out_relation->same_workitem_write_mask));

  loom_value_id_t same_category = LOOM_VALUE_ID_INVALID;
  loom_value_id_t conflict_kind = LOOM_VALUE_ID_INVALID;
  if (plan->shadow_access_kind == LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        prior_access_kind, read_kind, &same_category));
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
        prior_access_kind, read_kind, &conflict_kind));
  } else if (plan->shadow_access_kind ==
             LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        prior_access_kind, atomic_kind, &same_category));
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
        prior_access_kind, atomic_kind, &conflict_kind));
  } else {
    same_category = prior_write_like;
    conflict_kind = active_mask;
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, same_workitem,
      same_category, &out_relation->same_workitem_category_mask));
  loom_value_id_t different_workitem = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64, valid_mask,
      same_workitem, &different_workitem));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      different_workitem, conflict_kind, &out_relation->failure_mask));
  out_relation->access_kind = prior_access_kind;
  out_relation->valid_mask = valid_mask;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_add_publication_candidate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t condition_mask, bool second_slot,
    loom_value_id_t* inout_unselected_mask, loom_value_id_t* inout_publish_mask,
    loom_value_id_t* inout_second_slot_mask) {
  loom_value_id_t candidate_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, condition_mask,
      *inout_unselected_mask, &candidate_mask));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
      *inout_publish_mask, candidate_mask, inout_publish_mask));
  if (second_slot) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
        *inout_second_slot_mask, candidate_mask, inout_second_slot_mask));
  }
  return loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
      *inout_unselected_mask, candidate_mask, inout_unselected_mask);
}

static iree_status_t loom_amdgpu_sanitizer_race_select_publication(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_observation_plan_t* plan,
    loom_value_id_t active_mask,
    const loom_amdgpu_sanitizer_race_shadow_entry_t histories[2],
    const loom_amdgpu_sanitizer_race_history_relation_t relations[2],
    loom_amdgpu_sanitizer_race_publication_t* out_publication) {
  *out_publication = (loom_amdgpu_sanitizer_race_publication_t){0};
  loom_value_id_t zero_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64, active_mask,
      active_mask, &zero_mask));
  loom_value_id_t unselected_mask = active_mask;
  loom_value_id_t publish_mask = zero_mask;
  loom_value_id_t second_slot_mask = zero_mask;

  const uint32_t current_access_kind_bit = 1u << plan->shadow_access_kind;
  for (uint32_t i = 0; i < LOOM_AMDGPU_SANITIZER_RACE_HISTORY_RULE_COUNT; ++i) {
    const loom_amdgpu_sanitizer_race_history_rule_t* rule =
        &loom_amdgpu_sanitizer_race_history_rules[i];
    if ((rule->current_access_kind_mask & current_access_kind_bit) == 0) {
      continue;
    }
    loom_value_id_t condition_mask = LOOM_VALUE_ID_INVALID;
    switch (rule->condition) {
      case LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_SAME_WORKITEM_CATEGORY:
        condition_mask = relations[rule->condition_history_index]
                             .same_workitem_category_mask;
        break;
      case LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_INVALID: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
            active_mask, relations[rule->condition_history_index].valid_mask,
            &condition_mask));
        break;
      }
      case LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_CURRENT_WRITE_LIKE:
        condition_mask = active_mask;
        break;
      case LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_SAME_WORKITEM_WRITE:
        condition_mask =
            relations[rule->condition_history_index].same_workitem_write_mask;
        break;
      default:
        IREE_ASSERT_UNREACHABLE("verified race history publication rule");
        break;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_add_publication_candidate(
        context, source_op, condition_mask,
        /*second_slot=*/rule->selected_history_index == 1u, &unselected_mask,
        &publish_mask, &second_slot_mask));
  }

  loom_amdgpu_sanitizer_race_shadow_entry_t expected = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_shadow_entry_select(
      context, source_op, &histories[0], &histories[1], second_slot_mask,
      &expected));
  *out_publication = (loom_amdgpu_sanitizer_race_publication_t){
      .publish_mask = publish_mask,
      .second_slot_mask = second_slot_mask,
      .expected = expected,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_report(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_observation_plan_t* plan,
    const loom_amdgpu_sanitizer_race_access_range_t* access_range,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_shadow_address_t* shadow_address,
    loom_value_id_t current_workitem_linear_id,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* observed_entry,
    loom_value_id_t prior_access_kind,
    loom_amdgpu_feedback_packet_source_t* out_source,
    loom_amdgpu_sanitizer_race_report_t* out_report) {
  *out_source = (loom_amdgpu_feedback_packet_source_t){0};
  *out_report = (loom_amdgpu_sanitizer_race_report_t){0};

  const loom_amdgpu_sanitizer_race_report_coordinates_t* coordinates = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_report_coordinates(context, &coordinates));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, 0, &zero));

  loom_value_id_t prior_atomic_kind = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC,
      &prior_atomic_kind));
  loom_value_id_t prior_atomic_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      prior_access_kind, prior_atomic_kind, &prior_atomic_mask));

  const uint32_t static_flags =
      LOOM_AMDGPU_TSAN_REPORT_FLAG_CURRENT_WORKITEM_LINEAR |
      LOOM_AMDGPU_TSAN_REPORT_FLAG_PRIOR_WORKITEM_LINEAR |
      (plan->atomic ? LOOM_AMDGPU_TSAN_REPORT_FLAG_CURRENT_ATOMIC : 0u);
  loom_value_id_t base_flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, static_flags, &base_flags));
  loom_value_id_t atomic_flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op,
      static_flags | LOOM_AMDGPU_TSAN_REPORT_FLAG_PRIOR_ATOMIC, &atomic_flags));
  loom_value_id_t flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_select(
      context, source_op, base_flags, atomic_flags, prior_atomic_mask, &flags));

  out_source->dispatch_ptr = coordinates->dispatch_ptr;
  out_source->workgroup_id_x =
      coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_X];
  out_source->workitem_id_x = current_workitem_linear_id;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, LOOM_AMDGPU_TSAN_CHECK_KIND_DATA_RACE,
      &out_report->check_kind));
  out_report->flags = flags;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, LOOM_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP,
      &out_report->memory_space));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, plan->report_access_kind,
      &out_report->current_access_kind));
  out_report->prior_access_kind = prior_access_kind;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, access_range->byte_length, &out_report->access_size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_constant_u64(
      context, source_op, plan->site_id, &out_report->current_site_id));
  loom_value_id_t prior_site_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_decode_shadow_site_id(
      context, source_op, observed_entry->value, &prior_site_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_from_u32(
      context, source_op, prior_site_id, &out_report->prior_site_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_from_u32(
      context, source_op, access_range->memory_byte_offset,
      &out_report->memory_address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_shadow_report_address(
      context, source_op, config, shadow_address, &out_report->shadow_address));
  out_report->shadow_value = observed_entry->value;
  loom_value_id_t prior_workitem = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_decode_shadow_workitem(
      context, source_op, observed_entry->value, &prior_workitem));
  out_report->current_workgroup_id_x =
      coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_X];
  out_report->current_workgroup_id_y =
      coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_Y];
  out_report->current_workgroup_id_z =
      coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_Z];
  out_report->current_workitem_id_x = current_workitem_linear_id;
  out_report->current_workitem_id_y = zero;
  out_report->current_workitem_id_z = zero;
  out_report->prior_workgroup_id_x =
      coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_X];
  out_report->prior_workgroup_id_y =
      coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_Y];
  out_report->prior_workgroup_id_z =
      coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_Z];
  out_report->prior_workitem_id_x = prior_workitem;
  out_report->prior_workitem_id_y = zero;
  out_report->prior_workitem_id_z = zero;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_report_history_conflicts(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_observation_plan_t* plan,
    const loom_amdgpu_sanitizer_race_access_range_t* access_range,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_shadow_address_t* shadow_address,
    loom_value_id_t workitem_linear_id,
    const loom_amdgpu_sanitizer_race_shadow_entry_t histories[2],
    const loom_amdgpu_sanitizer_race_history_relation_t relations[2]) {
  loom_value_id_t failure_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
      relations[0].failure_mask, relations[1].failure_mask, &failure_mask));
  loom_amdgpu_sanitizer_race_shadow_entry_t failed_history = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_shadow_entry_select(
      context, source_op, &histories[1], &histories[0],
      relations[0].failure_mask, &failed_history));
  loom_value_id_t failed_access_kind = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_select(
      context, source_op, relations[1].access_kind, relations[0].access_kind,
      relations[0].failure_mask, &failed_access_kind));

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_sanitizer_reporting_mode_t reporting_mode =
      loom_low_lower_context_sanitizer_reporting_mode(context);
  switch (reporting_mode) {
    case LOOM_SANITIZER_REPORTING_MODE_TRAP: {
      loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
      const loom_amdgpu_sanitizer_trap_island_t* island = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_trap_island(
          context, source_op->location, &island));
      return loom_amdgpu_build_sanitizer_trap_failure_mask_branch_to_island(
          builder, descriptor_set, island, failure_mask, source_op->location,
          &branch);
    }
    case LOOM_SANITIZER_REPORTING_MODE_DEFAULT:
    case LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY: {
      loom_amdgpu_sanitizer_race_report_failure_branch_t branch = {0};
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_build_sanitizer_race_report_failure_mask_split(
              builder, descriptor_set, failure_mask, source_op->location,
              &branch));

      loom_amdgpu_feedback_packet_source_t source = {0};
      loom_amdgpu_sanitizer_race_report_t report = {0};
      loom_value_id_t report_workitem_linear_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
          context, source_op, workitem_linear_id, &report_workitem_linear_id));
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_report(
          context, source_op, plan, access_range, config, shadow_address,
          report_workitem_linear_id, &failed_history, failed_access_kind,
          &source, &report));
      const loom_amdgpu_sanitizer_race_report_island_t* island = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_report_island(
          context, source_op->location, &island));
      IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_race_report_branch(
          builder, descriptor_set, island, &source, &report,
          source_op->location));
      loom_builder_set_block(builder, branch.continuation_block);
      return iree_ok_status();
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported AMDGPU sanitizer reporting mode");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_sanitizer_race_build_history_equal_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* lhs,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* rhs,
    loom_value_id_t active_mask, loom_value_id_t* out_equal_mask) {
  *out_equal_mask = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_equal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, lhs->low,
      rhs->low, &low_equal));
  loom_value_id_t high_equal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, lhs->high,
      rhs->high, &high_equal));
  loom_value_id_t equal_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, low_equal,
      high_equal, &equal_mask));
  return loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, equal_mask,
      active_mask, out_equal_mask);
}

static iree_status_t loom_amdgpu_sanitizer_race_load_histories(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    loom_value_id_t first_history_offset, loom_value_id_t second_history_offset,
    loom_amdgpu_sanitizer_race_shadow_entry_t out_histories[2]) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_value_id_t offsets[2] = {first_history_offset,
                                      second_history_offset};
  for (uint32_t i = 0; i < 2; ++i) {
    loom_value_id_t history_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_global_load_b64(
        builder, descriptor_set, config->shadow_base, offsets[i],
        LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_ACQUIRE, source_op->location,
        &history_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_unpack_shadow_entry(
        context, source_op, history_value, &out_histories[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_emit_observation(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_observation_plan_t* plan,
    const loom_amdgpu_sanitizer_race_access_range_t* access_range,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_shadow_address_t* shadow_address,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* current_entry,
    loom_value_id_t workitem_linear_id) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);

  loom_value_id_t shadow_entry_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_shadow_entry_offset(
      context, source_op, shadow_address, &shadow_entry_offset));
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t second_history_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      shadow_entry_offset, LOOM_AMDGPU_TSAN_SHADOW_HISTORY_BYTE_LENGTH,
      vgpr_type, &second_history_offset));

  loom_value_id_t entry_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_read(
      builder, descriptor_set, source_op->location, &entry_exec));

  // A compare/exchange loser retries from a fresh two-history snapshot. Two
  // attempts suffice: every compatible winner either represents this access
  // already or fills one of the two summary roles. Keep the bounded retry in
  // Low CFG so the ordinary uncontended path emits and executes one body.
  loom_block_t* entry_block = builder->ip.block;
  loom_block_t* attempt_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, entry_block->parent_region,
      (uint16_t)(entry_block->region_index + 1), &attempt_block));
  loom_block_t* retry_check_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, attempt_block->parent_region,
      (uint16_t)(attempt_block->region_index + 1), &retry_check_block));
  loom_block_t* retry_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, retry_check_block->parent_region,
      (uint16_t)(retry_check_block->region_index + 1), &retry_block));
  loom_block_t* exit_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, retry_block->parent_region,
      (uint16_t)(retry_block->region_index + 1), &exit_block));

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t pending_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(builder, attempt_block,
                                                     mask_type, &pending_mask));
  loom_value_id_t attempt_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, attempt_block, sgpr_type, &attempt_index));

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, sgpr_type,
      &zero));
  const loom_value_id_t initial_args[] = {entry_exec, zero};
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, attempt_block, initial_args,
                                         IREE_ARRAYSIZE(initial_args),
                                         source_op->location, &branch_op));

  loom_builder_set_block(builder, attempt_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_restore(
      builder, descriptor_set, pending_mask, source_op->location));

  loom_amdgpu_sanitizer_race_shadow_entry_t histories[2] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_load_histories(
      context, source_op, config, shadow_entry_offset, second_history_offset,
      histories));
  loom_amdgpu_sanitizer_race_history_relation_t relations[2] = {0};
  for (uint32_t i = 0; i < 2; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_history_relation(
        context, source_op, plan, current_entry, &histories[i], pending_mask,
        &relations[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_report_history_conflicts(
      context, source_op, plan, access_range, config, shadow_address,
      workitem_linear_id, histories, relations));

  loom_amdgpu_sanitizer_race_publication_t publication = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_select_publication(
      context, source_op, plan, pending_mask, histories, relations,
      &publication));
  loom_value_id_t selected_history_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_select(
      context, source_op, shadow_entry_offset, second_history_offset,
      publication.second_slot_mask, &selected_history_offset));

  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_restore(
      builder, descriptor_set, publication.publish_mask, source_op->location));
  loom_value_id_t observed_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_build_global_compare_exchange_u64_acq_rel(
          builder, descriptor_set, config->shadow_base, selected_history_offset,
          publication.expected.value, current_entry->value, source_op->location,
          &observed_value));
  loom_amdgpu_sanitizer_race_shadow_entry_t observed = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_unpack_shadow_entry(
      context, source_op, observed_value, &observed));
  loom_value_id_t published_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_history_equal_mask(
      context, source_op, &observed, &publication.expected,
      publication.publish_mask, &published_mask));

  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_restore(
      builder, descriptor_set, pending_mask, source_op->location));
  loom_amdgpu_sanitizer_race_shadow_entry_t post_histories[2] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_load_histories(
      context, source_op, config, shadow_entry_offset, second_history_offset,
      post_histories));
  loom_amdgpu_sanitizer_race_history_relation_t post_relations[2] = {0};
  for (uint32_t i = 0; i < 2; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_history_relation(
        context, source_op, plan, current_entry, &post_histories[i],
        pending_mask, &post_relations[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_report_history_conflicts(
      context, source_op, plan, access_range, config, shadow_address,
      workitem_linear_id, post_histories, post_relations));

  loom_value_id_t retry_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
      publication.publish_mask, published_mask, &retry_mask));
  loom_value_id_t has_retry_lanes = LOOM_VALUE_ID_INVALID;
  const uint32_t wavefront_size =
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
      context, source_op, retry_mask, wavefront_size, &has_retry_lanes));
  loom_op_t* cond_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_cond_br_build(builder, has_retry_lanes, retry_check_block,
                             exit_block, source_op->location, &cond_branch_op));

  loom_builder_set_block(builder, retry_check_block);
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_value_id_t compare_operands[] = {attempt_index, zero};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32,
      compare_operands, IREE_ARRAYSIZE(compare_operands),
      loom_named_attr_slice_empty(), &scc_type, 1, &compare_op));
  const loom_value_id_t can_retry =
      loom_value_slice_get(loom_low_op_results(compare_op), 0);
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, can_retry, retry_block,
                                              exit_block, source_op->location,
                                              &cond_branch_op));

  loom_builder_set_block(builder, retry_block);
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 1, sgpr_type,
      &one));
  loom_value_id_t next_attempt_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, attempt_index,
      one, sgpr_type, &next_attempt_index));
  const loom_value_id_t retry_args[] = {retry_mask, next_attempt_index};
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, attempt_block, retry_args,
                                         IREE_ARRAYSIZE(retry_args),
                                         source_op->location, &branch_op));

  loom_builder_set_block(builder, exit_block);
  return loom_amdgpu_sanitizer_race_build_exec_restore(
      builder, descriptor_set, entry_exec, source_op->location);
}

static iree_status_t loom_amdgpu_sanitizer_race_emit_access_range(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_observation_plan_t* plan,
    const loom_amdgpu_sanitizer_race_access_range_t* access_range,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    loom_value_id_t workgroup_shadow_record_offset,
    loom_value_id_t workgroup_shadow_history_offset,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* current_entry,
    loom_value_id_t workitem_linear_id) {
  IREE_ASSERT_NE(access_range->byte_length, 0u);

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  const loom_value_id_t first_entry_index = access_range->memory_byte_offset;
  loom_amdgpu_sanitizer_race_shadow_address_t shadow_address = {0};
  if (access_range->byte_length == 1) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_sanitizer_race_build_shadow_entry_address_from_index(
            context, source_op, workgroup_shadow_record_offset,
            workgroup_shadow_history_offset, first_entry_index,
            &shadow_address));
    return loom_amdgpu_sanitizer_race_emit_observation(
        context, source_op, plan, access_range, config, &shadow_address,
        current_entry, workitem_linear_id);
  }

  loom_value_id_t last_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      access_range->memory_byte_offset, access_range->byte_length - 1,
      vgpr_type, &last_byte_offset));
  const loom_value_id_t last_entry_index = last_byte_offset;
  loom_value_id_t terminal_entry_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      last_entry_index, 1, vgpr_type, &terminal_entry_index));
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_block_t* entry_block = builder->ip.block;
  loom_block_t* loop_header_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, entry_block->parent_region,
      (uint16_t)(entry_block->region_index + 1), &loop_header_block));
  loom_block_t* loop_body_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, loop_header_block->parent_region,
      (uint16_t)(loop_header_block->region_index + 1), &loop_body_block));
  loom_block_t* exit_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, loop_body_block->parent_region,
      (uint16_t)(loop_body_block->region_index + 1), &exit_block));

  loom_value_id_t current_entry_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, loop_header_block, vgpr_type, &current_entry_index));
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, loop_header_block,
                                         &first_entry_index, /*args_count=*/1,
                                         source_op->location, &branch_op));

  loom_builder_set_block(builder, loop_header_block);
  loom_value_id_t remaining_lane_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      current_entry_index, terminal_entry_index, &remaining_lane_mask));
  loom_value_id_t incremented_entry_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      current_entry_index, 1, vgpr_type, &incremented_entry_index));
  loom_value_id_t next_entry_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_select(
      context, source_op, current_entry_index, incremented_entry_index,
      remaining_lane_mask, &next_entry_index));
  loom_value_id_t has_remaining_lanes = LOOM_VALUE_ID_INVALID;
  const uint32_t wavefront_size =
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
      context, source_op, remaining_lane_mask, wavefront_size,
      &has_remaining_lanes));
  loom_op_t* cond_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_cond_br_build(builder, has_remaining_lanes, loop_body_block,
                             exit_block, source_op->location, &cond_branch_op));

  loom_builder_set_block(builder, loop_body_block);
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_narrow_and_save(
      builder, descriptor_set, remaining_lane_mask, source_op->location,
      &saved_exec));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_build_shadow_entry_address_from_index(
          context, source_op, workgroup_shadow_record_offset,
          workgroup_shadow_history_offset, current_entry_index,
          &shadow_address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_emit_observation(
      context, source_op, plan, access_range, config, &shadow_address,
      current_entry, workitem_linear_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_restore(
      builder, descriptor_set, saved_exec, source_op->location));
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, loop_header_block,
                                         &next_entry_index, /*args_count=*/1,
                                         source_op->location, &branch_op));

  loom_builder_set_block(builder, exit_block);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_sanitizer_race_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_access_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);

  loom_amdgpu_sanitizer_race_config_values_t config = {0};
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_config_guard(
      context, source_op, &config, &continuation_block));

  loom_value_id_t workgroup_shadow_record_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_workgroup_shadow_record_offset(
          context, &workgroup_shadow_record_offset));
  loom_value_id_t workgroup_shadow_history_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_workgroup_shadow_history_offset(
          context, &workgroup_shadow_history_offset));

  loom_value_id_t memory_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_access_byte_offset(
      context, source_op, &plan->address, &memory_byte_offset));

  loom_value_id_t epoch = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_current_epoch(
      context, source_op, &config, workgroup_shadow_record_offset, &epoch));

  loom_value_id_t workitem_linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_workitem_linear_id(
      context, &workitem_linear_id));

  const loom_amdgpu_sanitizer_race_shadow_entry_base_t* shadow_entry_base =
      NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_shadow_entry_base(
      context, &shadow_entry_base));
  loom_amdgpu_sanitizer_race_shadow_entry_t current_entry = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_shadow_entry(
      context, source_op, plan->observation.shadow_access_kind,
      shadow_entry_base, epoch, plan->observation.site_id, &current_entry));

  const loom_amdgpu_sanitizer_race_access_range_t access_range = {
      .memory_byte_offset = memory_byte_offset,
      .byte_length = plan->access_size,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_establish_access_lifetime(
      context, source_op, &config, workgroup_shadow_record_offset,
      workgroup_shadow_history_offset, access_range.memory_byte_offset,
      access_range.byte_length));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_emit_access_range(
      context, source_op, &plan->observation, &access_range, &config,
      workgroup_shadow_record_offset, workgroup_shadow_history_offset,
      &current_entry, workitem_linear_id));

  return loom_amdgpu_sanitizer_race_branch_to_continuation(
      builder, continuation_block, source_op->location);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_fragment_participant_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* fragment_plan,
    const loom_amdgpu_matrix_fragment_lane_ids_t* lane_ids,
    loom_amdgpu_fragment_memory_issued_access_flags_t flags,
    loom_value_id_t entry_exec, loom_type_t vgpr_type,
    loom_value_id_t* inout_publishing_lane_mask,
    loom_value_id_t* out_participant_mask) {
  *out_participant_mask = entry_exec;
  if (iree_any_bit_set(
          flags,
          LOOM_AMDGPU_FRAGMENT_MEMORY_ISSUED_ACCESS_FLAG_PUBLISHERS_ONLY)) {
    if (*inout_publishing_lane_mask == LOOM_VALUE_ID_INVALID) {
      loom_type_t mask_type = loom_type_none();
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_publishing_lane_mask(
              context, source_op, fragment_plan, lane_ids, vgpr_type, mask_type,
              inout_publishing_lane_mask));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        *out_participant_mask, *inout_publishing_lane_mask,
        out_participant_mask));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_emit_fragment_access_group(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_observation_plan_t* observation_plan,
    const loom_amdgpu_fragment_memory_plan_t* fragment_plan,
    const loom_amdgpu_sanitizer_race_fragment_access_t* accesses,
    uint16_t access_count,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    loom_value_id_t workgroup_shadow_record_offset,
    loom_value_id_t workgroup_shadow_history_offset,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* current_entry,
    loom_value_id_t workitem_linear_id,
    const loom_amdgpu_matrix_fragment_lane_ids_t* lane_ids) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t entry_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_read(
      builder, descriptor_set, source_op->location, &entry_exec));

  loom_value_id_t publishing_lane_mask = LOOM_VALUE_ID_INVALID;
  if (access_count == 1u) {
    loom_amdgpu_matrix_fragment_lane_ids_t group_lane_ids = *lane_ids;
    loom_amdgpu_fragment_memory_address_state_t address_state;
    IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fragment_memory_address_state(
        context, source_op, fragment_plan, &group_lane_ids, vgpr_type,
        &address_state));
    loom_value_id_t memory_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_byte_offset(
        context, source_op, fragment_plan, accesses[0].access.register_index,
        accesses[0].access.element_index, accesses[0].access.descriptor_ref,
        &address_state, vgpr_type, &memory_byte_offset));
    loom_value_id_t participant_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_sanitizer_race_build_fragment_participant_mask(
            context, source_op, fragment_plan, &group_lane_ids,
            accesses[0].access.flags, entry_exec, vgpr_type,
            &publishing_lane_mask, &participant_mask));
    loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_narrow_and_save(
        builder, descriptor_set, participant_mask, source_op->location,
        &saved_exec));
    const loom_amdgpu_sanitizer_race_access_range_t access_range = {
        .memory_byte_offset = memory_byte_offset,
        .byte_length = accesses[0].byte_length,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_establish_access_lifetime(
        context, source_op, config, workgroup_shadow_record_offset,
        workgroup_shadow_history_offset, access_range.memory_byte_offset,
        access_range.byte_length));
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_emit_access_range(
        context, source_op, observation_plan, &access_range, config,
        workgroup_shadow_record_offset, workgroup_shadow_history_offset,
        current_entry, workitem_linear_id));
    return loom_amdgpu_sanitizer_race_build_exec_restore(
        builder, descriptor_set, saved_exec, source_op->location);
  }

  loom_value_id_t initial_access_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, 0u, &initial_access_index));
  loom_value_id_t terminal_access_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, access_count, &terminal_access_index));

  loom_block_t* entry_block = builder->ip.block;
  loom_block_t* loop_header_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, entry_block->parent_region,
      (uint16_t)(entry_block->region_index + 1), &loop_header_block));
  loom_block_t* loop_body_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, loop_header_block->parent_region,
      (uint16_t)(loop_header_block->region_index + 1), &loop_body_block));
  loom_block_t* exit_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, loop_body_block->parent_region,
      (uint16_t)(loop_body_block->region_index + 1), &exit_block));

  loom_value_id_t current_access_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, loop_header_block, vgpr_type, &current_access_index));
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, loop_header_block, &initial_access_index,
                        /*args_count=*/1, source_op->location, &branch_op));

  loom_builder_set_block(builder, loop_header_block);
  loom_value_id_t remaining_lane_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      current_access_index, terminal_access_index, &remaining_lane_mask));
  loom_value_id_t has_remaining_lanes = LOOM_VALUE_ID_INVALID;
  const uint32_t wavefront_size =
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
      context, source_op, remaining_lane_mask, wavefront_size,
      &has_remaining_lanes));
  loom_op_t* cond_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_cond_br_build(builder, has_remaining_lanes, loop_body_block,
                             exit_block, source_op->location, &cond_branch_op));

  loom_builder_set_block(builder, loop_body_block);
  loom_amdgpu_matrix_fragment_lane_ids_t group_lane_ids = *lane_ids;
  loom_amdgpu_fragment_memory_address_state_t address_state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fragment_memory_address_state(
      context, source_op, fragment_plan, &group_lane_ids, vgpr_type,
      &address_state));
  loom_value_id_t selected_byte_offset = LOOM_VALUE_ID_INVALID;
  loom_value_id_t selected_participant_mask = LOOM_VALUE_ID_INVALID;
  for (uint16_t i = 0; i < access_count; ++i) {
    loom_value_id_t candidate_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_byte_offset(
        context, source_op, fragment_plan, accesses[i].access.register_index,
        accesses[i].access.element_index, accesses[i].access.descriptor_ref,
        &address_state, vgpr_type, &candidate_byte_offset));
    loom_value_id_t candidate_participant_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_sanitizer_race_build_fragment_participant_mask(
            context, source_op, fragment_plan, &group_lane_ids,
            accesses[i].access.flags, entry_exec, vgpr_type,
            &publishing_lane_mask, &candidate_participant_mask));
    if (i == 0u) {
      selected_byte_offset = candidate_byte_offset;
      selected_participant_mask = candidate_participant_mask;
      continue;
    }
    loom_value_id_t candidate_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
        context, source_op, i, &candidate_index));
    loom_value_id_t candidate_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        current_access_index, candidate_index, &candidate_mask));
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_select(
        context, source_op, selected_byte_offset, candidate_byte_offset,
        candidate_mask, &selected_byte_offset));
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_select(
        context, source_op, selected_participant_mask,
        candidate_participant_mask, candidate_mask,
        &selected_participant_mask));
  }

  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_narrow_and_save(
      builder, descriptor_set, selected_participant_mask, source_op->location,
      &saved_exec));
  const loom_amdgpu_sanitizer_race_access_range_t access_range = {
      .memory_byte_offset = selected_byte_offset,
      .byte_length = accesses[0].byte_length,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_establish_access_lifetime(
      context, source_op, config, workgroup_shadow_record_offset,
      workgroup_shadow_history_offset, access_range.memory_byte_offset,
      access_range.byte_length));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_emit_access_range(
      context, source_op, observation_plan, &access_range, config,
      workgroup_shadow_record_offset, workgroup_shadow_history_offset,
      current_entry, workitem_linear_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_restore(
      builder, descriptor_set, saved_exec, source_op->location));

  loom_value_id_t next_access_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      current_access_index, 1u, vgpr_type, &next_access_index));
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, loop_header_block,
                                         &next_access_index, /*args_count=*/1,
                                         source_op->location, &branch_op));

  loom_builder_set_block(builder, exit_block);
  return loom_amdgpu_sanitizer_race_build_exec_restore(
      builder, descriptor_set, entry_exec, source_op->location);
}

iree_status_t loom_amdgpu_lower_sanitizer_race_fragment_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_fragment_access_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_amdgpu_fragment_memory_plan_t* fragment_plan =
      &plan->fragment_memory;

  loom_amdgpu_sanitizer_race_config_values_t config = {0};
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_config_guard(
      context, source_op, &config, &continuation_block));

  loom_value_id_t workgroup_shadow_record_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_workgroup_shadow_record_offset(
          context, &workgroup_shadow_record_offset));
  loom_value_id_t workgroup_shadow_history_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_workgroup_shadow_history_offset(
          context, &workgroup_shadow_history_offset));
  loom_value_id_t epoch = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_current_epoch(
      context, source_op, &config, workgroup_shadow_record_offset, &epoch));
  loom_value_id_t workitem_linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_workitem_linear_id(
      context, &workitem_linear_id));

  const loom_amdgpu_sanitizer_race_shadow_entry_base_t* shadow_entry_base =
      NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_shadow_entry_base(
      context, &shadow_entry_base));
  loom_amdgpu_sanitizer_race_shadow_entry_t current_entry = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_shadow_entry(
      context, source_op, plan->observation.shadow_access_kind,
      shadow_entry_base, epoch, plan->observation.site_id, &current_entry));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_amdgpu_matrix_fragment_lane_ids_t lane_ids;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_matrix_fragment_lane_ids(
      context, source_op, fragment_plan->address_layout.primary_lane_divisor,
      vgpr_type, &lane_ids));
  loom_amdgpu_sanitizer_race_fragment_access_t
      accesses[LOOM_AMDGPU_SANITIZER_RACE_FRAGMENT_ACCESS_CAPACITY];
  uint16_t access_count = 0;
  for (uint16_t packet_index = 0; packet_index < fragment_plan->packet_count;
       ++packet_index) {
    const loom_amdgpu_fragment_memory_packet_plan_t* packet =
        &fragment_plan->packets[packet_index];
    loom_amdgpu_fragment_memory_issued_access_t issued_accesses
        [LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET];
    const uint16_t issued_access_count =
        loom_amdgpu_fragment_memory_query_issued_accesses(fragment_plan, packet,
                                                          issued_accesses);
    for (uint16_t access_index = 0; access_index < issued_access_count;
         ++access_index) {
      IREE_ASSERT_LT(access_count,
                     LOOM_AMDGPU_SANITIZER_RACE_FRAGMENT_ACCESS_CAPACITY);
      accesses[access_count++] = (loom_amdgpu_sanitizer_race_fragment_access_t){
          .access = issued_accesses[access_index],
          .byte_length = (uint32_t)issued_accesses[access_index].element_count *
                         fragment_plan->element_byte_count,
      };
    }
  }

  for (uint16_t group_begin = 0; group_begin < access_count;) {
    uint16_t group_end = (uint16_t)(group_begin + 1u);
    while (group_end < access_count && accesses[group_end].byte_length ==
                                           accesses[group_begin].byte_length) {
      ++group_end;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_emit_fragment_access_group(
        context, source_op, &plan->observation, fragment_plan,
        &accesses[group_begin], (uint16_t)(group_end - group_begin), &config,
        workgroup_shadow_record_offset, workgroup_shadow_history_offset,
        &current_entry, workitem_linear_id, &lane_ids));
    group_begin = group_end;
  }

  return loom_amdgpu_sanitizer_race_branch_to_continuation(
      builder, continuation_block, source_op->location);
}

iree_status_t loom_amdgpu_lower_sanitizer_race_sync(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_sync_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_amdgpu_sanitizer_race_config_values_t config = {0};
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_config_guard(
      context, source_op, &config, &continuation_block));

  loom_value_id_t workgroup_shadow_record_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_workgroup_shadow_record_offset(
          context, &workgroup_shadow_record_offset));
  loom_value_id_t workgroup_chunk_state_bitmap_size = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_workgroup_chunk_state_bitmap_size(
          context, &workgroup_chunk_state_bitmap_size));
  loom_value_id_t workitem_linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_workitem_linear_id(
      context, &workitem_linear_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_advance_barrier_epoch(
      context, source_op, &config, workgroup_shadow_record_offset,
      workgroup_chunk_state_bitmap_size, workitem_linear_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_workgroup_barrier_plan(
      context, source_op, &plan->barrier));

  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_branch_to_continuation(
      builder, continuation_block, source_op->location));
  return iree_ok_status();
}
