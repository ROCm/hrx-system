// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_lifetime.h"

#include "iree/base/internal/math.h"
#include "loom/codegen/low/builder.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/memory_packet.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_state.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/value/integer64.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
    loom_builder_t* builder, loom_block_t* block, loom_block_t** out_block) {
  return loom_region_insert_block(builder->module, block->parent_region,
                                  (uint16_t)(block->region_index + 1u),
                                  out_block);
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
                                    vgpr_type, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_vgpr_cmp_mask(
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

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_sgpr64_equal_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_U64, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &scc_type, 1,
      &op));
  *out_scc = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_tags_match_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t observed_layout_tag, loom_value_id_t expected_layout_tag,
    loom_value_id_t observed_generation_tag,
    loom_value_id_t expected_generation_tag, loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;
  loom_value_id_t layout_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_sgpr64_equal_scc(
      context, source_op, observed_layout_tag, expected_layout_tag,
      &layout_scc));
  loom_value_id_t generation_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_sgpr64_equal_scc(
      context, source_op, observed_generation_tag, expected_generation_tag,
      &generation_scc));

  loom_value_id_t layout_word = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_scc_word(context, source_op, layout_scc, &layout_word));
  loom_value_id_t generation_word = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scc_word(
      context, source_op, generation_scc, &generation_word));
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t both_match = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32, layout_word,
      generation_word, sgpr_type, &both_match));
  return loom_amdgpu_emit_sgpr32_nonzero_scc(context, source_op, both_match,
                                             out_scc);
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_load_tags(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t record_base, loom_value_id_t* out_layout_tag,
    loom_value_id_t* out_generation_tag) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, record_base,
      LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_ACQUIRE, source_op->location,
      out_layout_tag));
  return loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, record_base,
      LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_GENERATION_TAG_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_ACQUIRE, source_op->location,
      out_generation_tag);
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_build_global_store(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value_unit_count,
    loom_value_id_t record_base, loom_value_id_t byte_offset,
    uint32_t immediate_offset, loom_value_id_t value, bool release,
    loom_location_id_t location) {
  loom_value_id_t vgpr_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, value, value_unit_count, location, &vgpr_value));

  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_offset_attr(
      builder, immediate_offset, &attrs[attr_count++]));
  if (release) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_system_memory_append_release_store_attrs_scoped(
            builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs,
            IREE_ARRAYSIZE(attrs), &attr_count));
  }
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/1,
                                               attrs, &attr_count);

  const loom_value_id_t operands[] = {byte_offset, vgpr_value, record_base};
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op);
}

static iree_status_t
loom_amdgpu_sanitizer_race_lifetime_build_exec_narrow_and_save(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lane_mask, loom_location_id_t location,
    loom_value_id_t* out_saved_exec) {
  *out_saved_exec = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &scc_type));
  const loom_type_t result_types[] = {mask_type, scc_type};
  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0, &lane_mask,
      /*operand_count=*/1, loom_named_attr_slice_empty(), result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op));
  *out_saved_exec = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_build_exec_restore(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t saved_exec, loom_location_id_t location) {
  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC);
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0, &saved_exec,
      /*operand_count=*/1, loom_named_attr_slice_empty(),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op);
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_build_exec_read(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location, loom_value_id_t* out_exec) {
  *out_exec = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));
  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      &mask_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op));
  *out_exec = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_mask_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  return loom_amdgpu_emit_binary(context, source_op, descriptor_ref, lhs, rhs,
                                 mask_type, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t condition_mask, loom_value_id_t* out_value) {
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

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_build_global_load_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &result_type));
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
      &result_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return loom_amdgpu_system_memory_build_acquire_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location);
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_build_global_or_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_value_id_t value, bool returning, loom_location_id_t location,
    loom_value_id_t* out_old_value) {
  if (out_old_value) {
    *out_old_value = LOOM_VALUE_ID_INVALID;
  }
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      returning ? LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_OR_B32_RTN_SADDR
                : LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_OR_B32_SADDR;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  loom_value_id_t value_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, value, /*expected_unit_count=*/1, location,
      &value_vgpr));
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_atomic_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {byte_offset, value_vgpr, base_address};
  loom_type_t result_type = loom_type_none();
  const loom_type_t* result_types = NULL;
  iree_host_size_t result_count = 0;
  if (returning) {
    IREE_RETURN_IF_ERROR(loom_low_build_register_type(
        descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &result_type));
    result_types = &result_type;
    result_count = 1;
  }
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      result_types, result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op));
  if (returning) {
    *out_old_value = loom_value_slice_get(loom_low_op_results(op), 0);
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_sanitizer_race_lifetime_build_global_add_returning(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t register_count,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_value_id_t value, loom_location_id_t location,
    loom_value_id_t* out_old_value) {
  *out_old_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  loom_value_id_t value_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, value, register_count, location, &value_vgpr));
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_atomic_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {byte_offset, value_vgpr, base_address};
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, register_count,
      &result_type));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &result_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op));
  *out_old_value = loom_value_slice_get(loom_low_op_results(op), 0);
  // The wrap test consumes the returned value. Drain the update itself before
  // any wave reaches the caller's workgroup barrier.
  return loom_amdgpu_system_memory_build_release_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location);
}

// Clears an aligned chunk-state bitmap with the active first wave. The caller
// owns convergence before and publication after the clear.
static iree_status_t
loom_amdgpu_sanitizer_race_lifetime_clear_chunk_state_bitmap(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t base_address, loom_value_id_t record_byte_offset,
    loom_value_id_t bitmap_byte_length, loom_value_id_t workitem_linear_id,
    loom_value_id_t zero) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_location_id_t location = source_op->location;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_type_t vgpr_x4_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 4, &vgpr_x4_type));
  const loom_value_id_t zero_parts[] = {zero, zero, zero, zero};
  loom_op_t* zero_concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, zero_parts, IREE_ARRAYSIZE(zero_parts),
                            vgpr_x4_type, location, &zero_concat_op));

  loom_value_id_t initial_bitmap_offset = LOOM_VALUE_ID_INVALID;
  const uint32_t bitmap_alignment_shift =
      (uint32_t)iree_math_count_trailing_zeros_u32(
          LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_BITMAP_ALIGNMENT);
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      bitmap_alignment_shift, workitem_linear_id, vgpr_type,
      &initial_bitmap_offset));

  loom_block_t* entry_block = builder->ip.block;
  loom_block_t* header_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, entry_block, &header_block));
  loom_block_t* body_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, header_block, &body_block));
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, body_block, &continuation_block));

  loom_value_id_t current_bitmap_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, header_block, vgpr_type, &current_bitmap_offset));
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, header_block, &initial_bitmap_offset,
                        /*args_count=*/1, location, &branch_op));

  loom_builder_set_block(builder, header_block);
  loom_value_id_t remaining_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
      current_bitmap_offset, bitmap_byte_length, &remaining_mask));
  const uint32_t wavefront_size =
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context));
  loom_value_id_t has_remaining_lanes_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
      context, source_op, remaining_mask, wavefront_size,
      &has_remaining_lanes_scc));
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, has_remaining_lanes_scc,
                                              body_block, continuation_block,
                                              location, &branch_op));

  loom_builder_set_block(builder, body_block);
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_lifetime_build_exec_narrow_and_save(
          builder, descriptor_set, remaining_mask, location, &saved_exec));
  loom_value_id_t store_byte_offset = current_bitmap_offset;
  if (record_byte_offset != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
        record_byte_offset, current_bitmap_offset, vgpr_type,
        &store_byte_offset));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B128_SADDR,
      /*value_unit_count=*/4, base_address, store_byte_offset,
      LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_CHUNK_STATE_OFFSET,
      loom_low_concat_result(zero_concat_op), /*release=*/false, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_exec_restore(
      builder, descriptor_set, saved_exec, location));

  const loom_target_workgroup_size_t* workgroup_size =
      loom_low_lower_context_workgroup_size(context);
  const uint32_t flat_workgroup_size =
      workgroup_size->x * workgroup_size->y * workgroup_size->z;
  const uint32_t first_wave_lane_count = flat_workgroup_size < wavefront_size
                                             ? flat_workgroup_size
                                             : wavefront_size;
  loom_value_id_t next_bitmap_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      current_bitmap_offset,
      first_wave_lane_count *
          LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_BITMAP_ALIGNMENT,
      vgpr_type, &next_bitmap_offset));
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, header_block, &next_bitmap_offset,
                        /*args_count=*/1, location, &branch_op));

  loom_builder_set_block(builder, continuation_block);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_build_expected_tags(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_shadow_layout_t* layout,
    loom_value_id_t dispatch_id, loom_value_id_t* out_layout_tag,
    loom_value_id_t* out_generation_tag) {
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));
  loom_value_id_t marker = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
      context, source_op, LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_MARKER,
      &marker));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
      layout->workgroup_shadow_stride, marker, sgpr_x2_type, out_layout_tag));

  loom_named_attr_t shift_attr = {0};
  iree_host_size_t shift_attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("shift"),
      LOOM_AMDGPU_TSAN_SHADOW_GENERATION_WINDOW_SHIFT, &shift_attr,
      /*attr_capacity=*/1, &shift_attr_count));
  loom_op_t* shift_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B64_RHS_INLINE,
      &dispatch_id, /*operand_count=*/1,
      loom_make_named_attr_slice(&shift_attr, shift_attr_count), &sgpr_x2_type,
      /*result_count=*/1, &shift_op));
  const loom_value_id_t generation_window =
      loom_value_slice_get(loom_low_op_results(shift_op), 0);
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_sgpr64_constant_u64(context, source_op, 1u, &one));
  return loom_amdgpu_emit_sgpr64_binary_carry(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32, generation_window, one,
      out_generation_tag);
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_build_record_base(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_shadow_layout_t* layout,
    loom_value_id_t* out_record_base) {
  return loom_amdgpu_emit_sgpr64_binary_carry(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32, config->shadow_base,
      layout->workgroup_shadow_offset, out_record_base);
}

iree_status_t loom_amdgpu_sanitizer_race_establish_workgroup_lifetime(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_shadow_layout_t* layout,
    loom_value_id_t workitem_linear_id, loom_value_id_t dispatch_id) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_location_id_t location = source_op->location;

  loom_value_id_t record_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_record_base(
      context, source_op, config, layout, &record_base));
  loom_value_id_t expected_layout_tag = LOOM_VALUE_ID_INVALID;
  loom_value_id_t expected_generation_tag = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_expected_tags(
      context, source_op, layout, dispatch_id, &expected_layout_tag,
      &expected_generation_tag));

  loom_value_id_t observed_layout_tag = LOOM_VALUE_ID_INVALID;
  loom_value_id_t observed_generation_tag = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_load_tags(
      context, source_op, record_base, &observed_layout_tag,
      &observed_generation_tag));
  loom_value_id_t tags_match_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_tags_match_scc(
      context, source_op, observed_layout_tag, expected_layout_tag,
      observed_generation_tag, expected_generation_tag, &tags_match_scc));

  loom_block_t* entry_block = builder->ip.block;
  loom_block_t* mismatch_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, entry_block, &mismatch_block));
  loom_block_t* initializer_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, mismatch_block, &initializer_block));
  loom_block_t* wait_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, initializer_block, &wait_block));
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, wait_block, &continuation_block));

  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_cond_br_build(builder, tags_match_scc, continuation_block,
                             mismatch_block, location, &branch_op));

  loom_builder_set_block(builder, mismatch_block);
  const uint32_t wavefront_size =
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context));
  loom_value_id_t wavefront_size_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_constant(
      context, source_op, wavefront_size, &wavefront_size_value));
  loom_value_id_t first_wave_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
      workitem_linear_id, wavefront_size_value, &first_wave_mask));
  loom_value_id_t is_first_wave_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
      context, source_op, first_wave_mask, wavefront_size, &is_first_wave_scc));
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, is_first_wave_scc,
                                              initializer_block, wait_block,
                                              location, &branch_op));

  loom_builder_set_block(builder, initializer_block);
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_constant(
      context, source_op, 0u, &zero));
  loom_value_id_t zero_u64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_vgpr64_constant_u64(context, source_op, 0u, &zero_u64));
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t state_bitmap_size_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, layout->workgroup_chunk_state_bitmap_size,
      /*index=*/0, sgpr_type, &state_bitmap_size_low));
  loom_value_id_t state_bitmap_size = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, state_bitmap_size_low, &state_bitmap_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_lifetime_clear_chunk_state_bitmap(
          context, source_op, record_base,
          /*record_byte_offset=*/LOOM_VALUE_ID_INVALID, state_bitmap_size,
          workitem_linear_id, zero));

  loom_type_t vgpr_x4_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 4, &vgpr_x4_type));
  loom_value_id_t leader_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      workitem_linear_id, zero, &leader_mask));
  loom_value_id_t leader_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_lifetime_build_exec_narrow_and_save(
          builder, descriptor_set, leader_mask, location, &leader_exec));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      /*value_unit_count=*/2, record_base, zero,
      LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_EPOCH_OFFSET, zero_u64,
      /*release=*/false, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      /*value_unit_count=*/2, record_base, zero,
      LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_RESERVED_OFFSET, zero_u64,
      /*release=*/false, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location));
  loom_value_id_t layout_tag_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, expected_layout_tag,
      /*expected_unit_count=*/2, location, &layout_tag_vgpr));
  loom_value_id_t generation_tag_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, expected_generation_tag,
      /*expected_unit_count=*/2, location, &generation_tag_vgpr));
  const loom_value_id_t tag_parts[] = {layout_tag_vgpr, generation_tag_vgpr};
  loom_op_t* tag_concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, tag_parts, IREE_ARRAYSIZE(tag_parts),
                            vgpr_x4_type, location, &tag_concat_op));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B128_SADDR,
      /*value_unit_count=*/4, record_base, zero,
      LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_OFFSET,
      loom_low_concat_result(tag_concat_op), /*release=*/true, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_exec_restore(
      builder, descriptor_set, leader_exec, location));
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, continuation_block,
                                         /*args=*/NULL, /*args_count=*/0,
                                         location, &branch_op));

  loom_builder_set_block(builder, wait_block);
  loom_value_id_t waited_layout_tag = LOOM_VALUE_ID_INVALID;
  loom_value_id_t waited_generation_tag = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_load_tags(
      context, source_op, record_base, &waited_layout_tag,
      &waited_generation_tag));
  loom_value_id_t wait_complete_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_tags_match_scc(
      context, source_op, waited_layout_tag, expected_layout_tag,
      waited_generation_tag, expected_generation_tag, &wait_complete_scc));
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, wait_complete_scc,
                                              continuation_block, wait_block,
                                              location, &branch_op));

  loom_builder_set_block(builder, continuation_block);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_advance_barrier_epoch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    loom_value_id_t workgroup_shadow_record_offset,
    loom_value_id_t workgroup_chunk_state_bitmap_size,
    loom_value_id_t workitem_linear_id) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_location_id_t location = source_op->location;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t record_offset_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, workgroup_shadow_record_offset, /*index=*/0,
      vgpr_type, &record_offset_low));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_constant(
      context, source_op, 0u, &zero));
  loom_value_id_t leader_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      workitem_linear_id, zero, &leader_mask));

  // Every wave executes the transition, but only the wave containing
  // workitem zero has a nonempty narrowed EXEC. The masked compare below
  // therefore leaves wraps_scc false in every other wave.
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_lifetime_build_exec_narrow_and_save(
          builder, descriptor_set, leader_mask, location, &saved_exec));
  loom_value_id_t old_epoch = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U32_RTN_SADDR)) {
    loom_value_id_t one = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_constant(
        context, source_op, 1u, &one));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_sanitizer_race_lifetime_build_global_add_returning(
            builder, descriptor_set,
            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U32_RTN_SADDR,
            /*register_count=*/1, config->shadow_base, record_offset_low, one,
            location, &old_epoch));
  } else {
    loom_value_id_t one = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_vgpr64_constant_u64(context, source_op, 1u, &one));
    loom_value_id_t old_epoch_u64 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_sanitizer_race_lifetime_build_global_add_returning(
            builder, descriptor_set,
            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_RTN_SADDR,
            /*register_count=*/2, config->shadow_base, record_offset_low, one,
            location, &old_epoch_u64));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, old_epoch_u64, /*index=*/0, vgpr_type, &old_epoch));
  }

  const uint32_t compact_epoch_mask =
      LOOM_AMDGPU_TSAN_SHADOW_EPOCH_CAPACITY - 1u;
  loom_value_id_t compact_epoch = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, old_epoch,
      compact_epoch_mask, vgpr_type, &compact_epoch));
  loom_value_id_t compact_epoch_mask_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_constant(
      context, source_op, compact_epoch_mask, &compact_epoch_mask_value));
  loom_value_id_t wrap_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      compact_epoch, compact_epoch_mask_value, &wrap_mask));
  loom_value_id_t wraps_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
      context, source_op, wrap_mask,
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context)),
      &wraps_scc));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_exec_restore(
      builder, descriptor_set, saved_exec, location));

  loom_block_t* entry_block = builder->ip.block;
  loom_block_t* wrap_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, entry_block, &wrap_block));
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, wrap_block, &continuation_block));
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, wraps_scc, wrap_block,
                                              continuation_block, location,
                                              &branch_op));

  loom_builder_set_block(builder, wrap_block);
  loom_value_id_t bitmap_byte_length = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, workgroup_chunk_state_bitmap_size,
      &bitmap_byte_length));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_lifetime_clear_chunk_state_bitmap(
          context, source_op, config->shadow_base, record_offset_low,
          bitmap_byte_length, workitem_linear_id, zero));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location));
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, continuation_block, /*args=*/NULL,
                        /*args_count=*/0, location, &branch_op));

  loom_builder_set_block(builder, continuation_block);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_lifetime_extract_chunk_state(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t state_word, loom_value_id_t state_shift,
    loom_value_id_t* out_state) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32, state_shift,
      state_word, vgpr_type, &shifted));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, shifted,
      (1u << LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_BIT_COUNT) - 1u, vgpr_type,
      out_state);
}

static iree_status_t loom_amdgpu_sanitizer_race_establish_one_chunk_lifetime(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    loom_value_id_t workgroup_shadow_record_offset,
    loom_value_id_t workgroup_shadow_history_offset,
    loom_value_id_t chunk_index) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_location_id_t location = source_op->location;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t state_word_index = LOOM_VALUE_ID_INVALID;
  const uint32_t chunks_per_state_word_shift =
      (uint32_t)iree_math_count_trailing_zeros_u32(
          LOOM_AMDGPU_TSAN_SHADOW_CHUNKS_PER_STATE_WORD);
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      chunks_per_state_word_shift, chunk_index, vgpr_type, &state_word_index));
  loom_value_id_t state_word_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      /*shift_amount=*/2u, state_word_index, vgpr_type,
      &state_word_byte_offset));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      state_word_byte_offset,
      LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_CHUNK_STATE_OFFSET, vgpr_type,
      &state_word_byte_offset));
  loom_value_id_t workgroup_shadow_record_offset_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, workgroup_shadow_record_offset, /*index=*/0,
      vgpr_type, &workgroup_shadow_record_offset_low));
  loom_value_id_t state_word_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      workgroup_shadow_record_offset_low, state_word_byte_offset, vgpr_type,
      &state_word_offset));

  loom_value_id_t chunk_in_word = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, chunk_index,
      LOOM_AMDGPU_TSAN_SHADOW_CHUNKS_PER_STATE_WORD - 1u, vgpr_type,
      &chunk_in_word));
  loom_value_id_t state_shift = LOOM_VALUE_ID_INVALID;
  const uint32_t state_bit_count_shift =
      (uint32_t)iree_math_count_trailing_zeros_u32(
          LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_BIT_COUNT);
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      state_bit_count_shift, chunk_in_word, vgpr_type, &state_shift));
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_constant(
      context, source_op, 1u, &one));
  loom_value_id_t claim_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32, state_shift,
      one, vgpr_type, &claim_mask));
  loom_value_id_t ready_mask_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      /*shift_amount=*/1u, claim_mask, vgpr_type, &ready_mask_value));

  loom_value_id_t chunk_history_relative_offset = LOOM_VALUE_ID_INVALID;
  const uint32_t chunk_history_byte_length =
      LOOM_AMDGPU_TSAN_SHADOW_CHUNK_APPLICATION_BYTE_LENGTH *
      LOOM_AMDGPU_TSAN_SHADOW_CELL_BYTE_LENGTH;
  const uint32_t chunk_history_byte_shift =
      (uint32_t)iree_math_count_trailing_zeros_u32(chunk_history_byte_length);
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      chunk_history_byte_shift, chunk_index, vgpr_type,
      &chunk_history_relative_offset));
  loom_value_id_t workgroup_shadow_history_offset_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, workgroup_shadow_history_offset, /*index=*/0,
      vgpr_type, &workgroup_shadow_history_offset_low));
  loom_value_id_t chunk_history_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      workgroup_shadow_history_offset_low, chunk_history_relative_offset,
      vgpr_type, &chunk_history_offset));

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_constant(
      context, source_op, 0u, &zero));
  loom_value_id_t ready_state = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_constant(
      context, source_op, LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_READY,
      &ready_state));
  loom_type_t vgpr_x4_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 4, &vgpr_x4_type));
  const loom_value_id_t zero_parts[] = {zero, zero, zero, zero};
  loom_op_t* zero_concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, zero_parts, IREE_ARRAYSIZE(zero_parts),
                            vgpr_x4_type, location, &zero_concat_op));

  loom_value_id_t entry_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_exec_read(
      builder, descriptor_set, location, &entry_exec));
  loom_block_t* entry_block = builder->ip.block;
  loom_block_t* check_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, entry_block, &check_block));
  loom_block_t* claim_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, check_block, &claim_block));
  loom_block_t* clear_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, claim_block, &clear_block));
  loom_block_t* clear_loop_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, clear_block, &clear_loop_block));
  loom_block_t* clear_continue_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, clear_loop_block, &clear_continue_block));
  loom_block_t* publish_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, clear_continue_block, &publish_block));
  loom_block_t* retry_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, publish_block, &retry_block));
  loom_block_t* exit_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, retry_block, &exit_block));

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  loom_value_id_t pending_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(builder, check_block,
                                                     mask_type, &pending_mask));
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, check_block, &entry_exec,
                                         /*args_count=*/1, location,
                                         &branch_op));

  loom_builder_set_block(builder, check_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_exec_restore(
      builder, descriptor_set, pending_mask, location));
  loom_value_id_t state_word = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_lifetime_build_global_load_b32(
          builder, descriptor_set, config->shadow_base, state_word_offset,
          location, &state_word));
  loom_value_id_t state = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_extract_chunk_state(
      context, source_op, state_word, state_shift, &state));
  loom_value_id_t ready_lanes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, state,
      ready_state, &ready_lanes));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, ready_lanes,
      pending_mask, &ready_lanes));
  loom_value_id_t remaining_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64, pending_mask,
      ready_lanes, &remaining_mask));
  const uint32_t wavefront_size =
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context));
  loom_value_id_t has_remaining_lanes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
      context, source_op, remaining_mask, wavefront_size,
      &has_remaining_lanes));
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, has_remaining_lanes,
                                              claim_block, exit_block, location,
                                              &branch_op));

  loom_builder_set_block(builder, claim_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_exec_restore(
      builder, descriptor_set, remaining_mask, location));
  loom_value_id_t uninitialized_lanes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, state, zero,
      &uninitialized_lanes));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_exec_restore(
      builder, descriptor_set, uninitialized_lanes, location));
  loom_value_id_t observed_word = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_global_or_b32(
      builder, descriptor_set, config->shadow_base, state_word_offset,
      claim_mask, /*returning=*/true, location, &observed_word));
  loom_value_id_t observed_state = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_extract_chunk_state(
      context, source_op, observed_word, state_shift, &observed_state));
  loom_value_id_t winner_lanes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      observed_state, zero, &winner_lanes));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, winner_lanes,
      uninitialized_lanes, &winner_lanes));
  loom_value_id_t has_winner_lanes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
      context, source_op, winner_lanes, wavefront_size, &has_winner_lanes));
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, has_winner_lanes,
                                              clear_block, retry_block,
                                              location, &branch_op));

  loom_builder_set_block(builder, clear_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_exec_restore(
      builder, descriptor_set, winner_lanes, location));
  loom_value_id_t clear_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, clear_loop_block, vgpr_type, &clear_byte_offset));
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, clear_loop_block, &zero,
                                         /*args_count=*/1, location,
                                         &branch_op));

  loom_builder_set_block(builder, clear_loop_block);
  loom_value_id_t clear_address_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      chunk_history_offset, clear_byte_offset, vgpr_type,
      &clear_address_offset));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B128_SADDR,
      /*value_unit_count=*/4, config->shadow_base, clear_address_offset,
      /*immediate_offset=*/0, loom_low_concat_result(zero_concat_op),
      /*release=*/false, location));
  loom_value_id_t next_clear_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      clear_byte_offset, LOOM_AMDGPU_TSAN_SHADOW_CELL_BYTE_LENGTH, vgpr_type,
      &next_clear_byte_offset));
  loom_value_id_t clear_limit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_constant(
      context, source_op, chunk_history_byte_length, &clear_limit));
  loom_value_id_t clear_remaining_lanes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      next_clear_byte_offset, clear_limit, &clear_remaining_lanes));
  loom_value_id_t has_clear_remaining_lanes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
      context, source_op, clear_remaining_lanes, wavefront_size,
      &has_clear_remaining_lanes));
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
      builder, has_clear_remaining_lanes, clear_continue_block, publish_block,
      location, &branch_op));

  loom_builder_set_block(builder, clear_continue_block);
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, clear_loop_block, &next_clear_byte_offset,
                        /*args_count=*/1, location, &branch_op));

  loom_builder_set_block(builder, publish_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_global_or_b32(
      builder, descriptor_set, config->shadow_base, state_word_offset,
      ready_mask_value, /*returning=*/false, location,
      /*out_old_value=*/NULL));
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, retry_block,
                                         /*args=*/NULL, /*args_count=*/0,
                                         location, &branch_op));

  loom_builder_set_block(builder, retry_block);
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, check_block, &remaining_mask,
                                         /*args_count=*/1, location,
                                         &branch_op));

  loom_builder_set_block(builder, exit_block);
  return loom_amdgpu_sanitizer_race_lifetime_build_exec_restore(
      builder, descriptor_set, entry_exec, location);
}

iree_status_t loom_amdgpu_sanitizer_race_establish_access_lifetime(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    loom_value_id_t workgroup_shadow_record_offset,
    loom_value_id_t workgroup_shadow_history_offset,
    loom_value_id_t memory_byte_offset, uint32_t byte_length) {
  IREE_ASSERT_NE(byte_length, 0u);
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_location_id_t location = source_op->location;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t first_chunk_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_CHUNK_APPLICATION_BYTE_SHIFT, memory_byte_offset,
      vgpr_type, &first_chunk_index));
  loom_value_id_t last_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      memory_byte_offset, byte_length - 1u, vgpr_type, &last_byte_offset));
  loom_value_id_t last_chunk_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_CHUNK_APPLICATION_BYTE_SHIFT, last_byte_offset,
      vgpr_type, &last_chunk_index));
  loom_value_id_t terminal_chunk_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      last_chunk_index, 1u, vgpr_type, &terminal_chunk_index));

  loom_value_id_t entry_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_exec_read(
      builder, descriptor_set, location, &entry_exec));
  loom_block_t* entry_block = builder->ip.block;
  loom_block_t* loop_header_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, entry_block, &loop_header_block));
  loom_block_t* loop_body_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, loop_header_block, &loop_body_block));
  loom_block_t* exit_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_insert_block_after(
      builder, loop_body_block, &exit_block));

  loom_value_id_t current_chunk_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, loop_header_block, vgpr_type, &current_chunk_index));
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, loop_header_block,
                                         &first_chunk_index, /*args_count=*/1,
                                         location, &branch_op));

  loom_builder_set_block(builder, loop_header_block);
  loom_value_id_t remaining_lanes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      current_chunk_index, terminal_chunk_index, &remaining_lanes));
  loom_value_id_t incremented_chunk_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      current_chunk_index, 1u, vgpr_type, &incremented_chunk_index));
  loom_value_id_t next_chunk_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_vgpr_u32_select(
      context, source_op, current_chunk_index, incremented_chunk_index,
      remaining_lanes, &next_chunk_index));
  loom_value_id_t has_remaining_lanes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
      context, source_op, remaining_lanes,
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context)),
      &has_remaining_lanes));
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, has_remaining_lanes,
                                              loop_body_block, exit_block,
                                              location, &branch_op));

  loom_builder_set_block(builder, loop_body_block);
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_lifetime_build_exec_narrow_and_save(
          builder, descriptor_set, remaining_lanes, location, &saved_exec));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_establish_one_chunk_lifetime(
      context, source_op, config, workgroup_shadow_record_offset,
      workgroup_shadow_history_offset, current_chunk_index));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lifetime_build_exec_restore(
      builder, descriptor_set, saved_exec, location));
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, loop_header_block,
                                         &next_chunk_index, /*args_count=*/1,
                                         location, &branch_op));

  loom_builder_set_block(builder, exit_block);
  return loom_amdgpu_sanitizer_race_lifetime_build_exec_restore(
      builder, descriptor_set, entry_exec, location);
}
