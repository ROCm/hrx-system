// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_state.h"

#include "loom/codegen/low/builder.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/lower/data_symbol.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/preamble.h"
#include "loom/target/arch/amdgpu/lower/sanitizer.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_lifetime.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_shadow_layout.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef struct loom_amdgpu_sanitizer_race_lower_state_t {
  // True once entry-block TSAN config values have been materialized.
  bool has_config_values;
  // Entry-dominating TSAN config values loaded from iree_tsan_config.
  loom_amdgpu_sanitizer_race_config_values_t config_values;
  // Entry-dominating TSAN config flags required by observation guards.
  loom_value_id_t required_config_flags;
  // Entry-dominating config.flags value masked by required_config_flags.
  loom_value_id_t active_config_flags;
  // True once entry-body TSAN instrumentation flags have been materialized.
  bool has_instrumentation_flags;
  // Entry-body guard flags. Zero disables observation lowering at runtime.
  loom_value_id_t instrumentation_flags;
  // True once entry-block launch topology values have been materialized.
  bool has_topology_values;
  // Entry-dominating flattened workitem id in the supported TSAN shadow range.
  loom_value_id_t workitem_linear_id;
  // True once entry-block report coordinates have been materialized.
  bool has_report_coordinates;
  // Entry-dominating VGPR coordinates carried by race reports.
  loom_amdgpu_sanitizer_race_report_coordinates_t report_coordinates;
  // True once the entry-body workgroup shadow record offset is materialized.
  bool has_workgroup_shadow_record_offset;
  // Entry-body byte offset of this workgroup shadow record from shadow_base.
  loom_value_id_t workgroup_shadow_record_offset;
  // True once the entry-body chunk-state bitmap size is materialized.
  bool has_workgroup_chunk_state_bitmap_size;
  // Entry-body aligned byte length of the chunk-state bitmap.
  loom_value_id_t workgroup_chunk_state_bitmap_size;
  // True once the entry-body history-array offset is materialized.
  bool has_workgroup_shadow_history_offset;
  // Entry-body byte offset of this workgroup history array from shadow_base.
  loom_value_id_t workgroup_shadow_history_offset;
  // True once entry-body shadow entry base bits have been materialized.
  bool has_shadow_entry_base;
  // Entry-body shadow entry bits shared by all access observations.
  loom_amdgpu_sanitizer_race_shadow_entry_base_t shadow_entry_base;
  // True once trap-mode TSAN failures have a shared cold island.
  bool has_trap_island;
  // Shared no-return cold island for trap-mode TSAN failures.
  loom_amdgpu_sanitizer_trap_island_t trap_island;
  // True once TSAN data-race failures have a shared cold report island.
  bool has_report_island;
  // Shared cold island for TSAN data-race reports.
  loom_amdgpu_sanitizer_race_report_island_t report_island;
} loom_amdgpu_sanitizer_race_lower_state_t;

static int loom_amdgpu_sanitizer_race_lower_state_key;

static iree_status_t loom_amdgpu_sanitizer_race_lower_state(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_race_lower_state_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_sanitizer_race_lower_state_key, sizeof(*state),
      (void**)&state));
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_config_values(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_symbol_ref_t config_symbol,
    loom_amdgpu_sanitizer_race_config_values_t* out_values) {
  *out_values = (loom_amdgpu_sanitizer_race_config_values_t){0};
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_location_id_t location = source_op->location;

  loom_amdgpu_sanitizer_race_config_values_t values = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_data_symbol_address(builder, descriptor_set,
                                            (loom_amdgpu_data_symbol_address_t){
                                                .symbol = config_symbol,
                                                .byte_offset = 0,
                                            },
                                            location, &values.address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b32(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_TSAN_CONFIG_FLAGS_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location, &values.flags));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_TSAN_CONFIG_SHADOW_BASE_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location, &values.shadow_base));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_TSAN_CONFIG_SHADOW_SIZE_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location, &values.shadow_size));
  *out_values = values;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_config_guard_values(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t config_flags, loom_value_id_t* out_required_flags,
    loom_value_id_t* out_active_flags) {
  *out_required_flags = LOOM_VALUE_ID_INVALID;
  *out_active_flags = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  const uint32_t required_flags = LOOM_AMDGPU_TSAN_CONFIG_FLAG_ENABLED |
                                  LOOM_AMDGPU_TSAN_CONFIG_FLAG_QUEUE_SHADOW;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, required_flags,
      sgpr_type, out_required_flags));
  return loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32, config_flags,
      *out_required_flags, sgpr_type, out_active_flags);
}

iree_status_t loom_amdgpu_sanitizer_race_get_config_values(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_race_config_values_t* out_config) {
  *out_config = (loom_amdgpu_sanitizer_race_config_values_t){0};
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(state->has_config_values,
              "AMDGPU TSAN config values were not materialized in entry setup");
  *out_config = state->config_values;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_get_workitem_linear_id(
    loom_low_lower_context_t* context, loom_value_id_t* out_linear_id) {
  *out_linear_id = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(
      state->has_topology_values,
      "AMDGPU TSAN topology values were not materialized in entry setup");
  *out_linear_id = state->workitem_linear_id;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_get_report_coordinates(
    loom_low_lower_context_t* context,
    const loom_amdgpu_sanitizer_race_report_coordinates_t** out_coordinates) {
  *out_coordinates = NULL;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(
      state->has_report_coordinates,
      "AMDGPU TSAN report coordinates were not materialized in entry setup");
  *out_coordinates = &state->report_coordinates;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_get_instrumentation_flags(
    loom_low_lower_context_t* context, loom_value_id_t* out_flags) {
  *out_flags = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(
      state->has_instrumentation_flags,
      "AMDGPU TSAN instrumentation flags were not materialized in entry setup");
  *out_flags = state->instrumentation_flags;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_get_workgroup_shadow_record_offset(
    loom_low_lower_context_t* context, loom_value_id_t* out_offset) {
  *out_offset = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(state->has_workgroup_shadow_record_offset,
              "AMDGPU TSAN workgroup shadow record offset was not "
              "materialized in entry setup");
  *out_offset = state->workgroup_shadow_record_offset;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_get_workgroup_chunk_state_bitmap_size(
    loom_low_lower_context_t* context, loom_value_id_t* out_size) {
  *out_size = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(state->has_workgroup_chunk_state_bitmap_size,
              "AMDGPU TSAN chunk-state bitmap size was not materialized in "
              "entry setup");
  *out_size = state->workgroup_chunk_state_bitmap_size;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_get_workgroup_shadow_history_offset(
    loom_low_lower_context_t* context, loom_value_id_t* out_offset) {
  *out_offset = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(state->has_workgroup_shadow_history_offset,
              "AMDGPU TSAN workgroup history offset was not materialized in "
              "entry setup");
  *out_offset = state->workgroup_shadow_history_offset;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_get_shadow_entry_base(
    loom_low_lower_context_t* context,
    const loom_amdgpu_sanitizer_race_shadow_entry_base_t** out_base) {
  *out_base = NULL;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(
      state->has_shadow_entry_base,
      "AMDGPU TSAN shadow entry base was not materialized in entry setup");
  *out_base = &state->shadow_entry_base;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_shadow_entry_base(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t workitem_linear_id, loom_value_id_t generation_low,
    loom_amdgpu_sanitizer_race_shadow_entry_base_t* out_base) {
  *out_base = (loom_amdgpu_sanitizer_race_shadow_entry_base_t){0};
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t workitem_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT, workitem_linear_id,
      vgpr_type, &workitem_bits));

  loom_value_id_t generation_low_part = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      generation_low,
      (1u << LOOM_AMDGPU_TSAN_SHADOW_ENTRY_LOW_GENERATION_BIT_COUNT) - 1u,
      vgpr_type, &generation_low_part));
  loom_value_id_t generation_low_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_SHIFT, generation_low_part,
      vgpr_type, &generation_low_bits));

  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, workitem_bits,
      generation_low_bits, vgpr_type, &low));

  loom_value_id_t generation_high_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_LOW_GENERATION_BIT_COUNT, generation_low,
      vgpr_type, &generation_high_bits));
  loom_value_id_t high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      generation_high_bits,
      (1u << LOOM_AMDGPU_TSAN_SHADOW_ENTRY_HIGH_GENERATION_BIT_COUNT) - 1u,
      vgpr_type, &high));

  *out_base = (loom_amdgpu_sanitizer_race_shadow_entry_base_t){
      .low = low,
      .high = high,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_emit_dispatch_generation_low(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_id, loom_value_id_t* out_generation_low) {
  *out_generation_low = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t dispatch_id_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, dispatch_id, 0, sgpr_type, &dispatch_id_low));
  return loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, dispatch_id_low, out_generation_low);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_required_config_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;

  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(state->has_config_values,
              "AMDGPU TSAN config guard values were not materialized in entry "
              "setup");

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_value_id_t operands[] = {state->active_config_flags,
                                      state->required_config_flags};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &scc_type, 1,
      &compare_op));
  *out_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_runtime_active_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;

  loom_value_id_t instrumentation_flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_instrumentation_flags(
      context, &instrumentation_flags));
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(state->has_config_values,
              "AMDGPU TSAN runtime guard values were not materialized in entry "
              "setup");

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_value_id_t operands[] = {instrumentation_flags,
                                      state->required_config_flags};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &scc_type, 1,
      &compare_op));
  *out_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_split_enabled_block(
    loom_builder_t* builder, loom_value_id_t enabled_scc,
    loom_location_id_t location, loom_block_t** out_enabled_block,
    loom_block_t** out_continuation_block) {
  *out_enabled_block = NULL;
  *out_continuation_block = NULL;
  IREE_ASSERT(
      builder->ip.before_op == NULL && builder->ip.block->parent_region != NULL,
      "AMDGPU sanitizer race lowering requires block-end insertion in a low "
      "region");
  loom_block_t* config_block = builder->ip.block;
  loom_block_t* enabled_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, config_block->parent_region,
      (uint16_t)(config_block->region_index + 1), &enabled_block));
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, enabled_block->parent_region,
      (uint16_t)(enabled_block->region_index + 1), &continuation_block));
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, enabled_scc,
                                              enabled_block, continuation_block,
                                              location, &branch_op));
  loom_builder_set_block(builder, enabled_block);
  *out_enabled_block = enabled_block;
  *out_continuation_block = continuation_block;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_branch_to_continuation(
    loom_builder_t* builder, loom_block_t* continuation_block,
    loom_location_id_t location) {
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, continuation_block,
                                         /*args=*/NULL, /*args_count=*/0,
                                         location, &branch_op));
  loom_builder_set_block(builder, continuation_block);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_branch_to_entry_body(
    loom_low_lower_context_t* context,
    const loom_low_lower_entry_interposition_t* interposition,
    const loom_value_id_t* target_args, loom_location_id_t location) {
  const uint16_t arg_count =
      interposition->forwarded_arg_count + interposition->target_arg_count;
  loom_value_id_t* args = NULL;
  if (arg_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, arg_count, sizeof(*args), (void**)&args));
  }
  for (uint16_t i = 0; i < interposition->forwarded_arg_count; ++i) {
    args[i] = interposition->forwarded_args[i];
  }
  for (uint16_t i = 0; i < interposition->target_arg_count; ++i) {
    args[interposition->forwarded_arg_count + i] = target_args[i];
  }
  loom_op_t* branch_op = NULL;
  return loom_low_br_build(loom_low_lower_context_builder(context),
                           interposition->body_block, args, arg_count, location,
                           &branch_op);
}

static iree_status_t loom_amdgpu_sanitizer_race_split_entry_guard_blocks(
    loom_builder_t* builder, loom_value_id_t enabled_scc,
    loom_location_id_t location, loom_block_t** out_enabled_block,
    loom_block_t** out_disabled_block) {
  if (out_enabled_block != NULL) {
    *out_enabled_block = NULL;
  }
  if (out_disabled_block != NULL) {
    *out_disabled_block = NULL;
  }
  IREE_ASSERT(
      builder->ip.before_op == NULL && builder->ip.block->parent_region != NULL,
      "AMDGPU sanitizer race entry setup requires block-end insertion in a low "
      "region");
  loom_block_t* setup_block = builder->ip.block;
  loom_block_t* enabled_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, setup_block->parent_region,
      (uint16_t)(setup_block->region_index + 1), &enabled_block));
  loom_block_t* disabled_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, enabled_block->parent_region,
      (uint16_t)(enabled_block->region_index + 1), &disabled_block));
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, enabled_scc,
                                              enabled_block, disabled_block,
                                              location, &branch_op));
  loom_builder_set_block(builder, enabled_block);
  if (out_enabled_block != NULL) {
    *out_enabled_block = enabled_block;
  }
  if (out_disabled_block != NULL) {
    *out_disabled_block = disabled_block;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_get_report_island(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_amdgpu_sanitizer_race_report_island_t** out_island) {
  *out_island = NULL;
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  if (!state->has_report_island) {
    loom_symbol_ref_t feedback_config_symbol = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_feedback_config_symbol(
        context, &feedback_config_symbol));
    loom_builder_ip_t saved_ip = loom_builder_save(builder);
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_race_report_island(
        builder, descriptor_set, saved_ip.block, feedback_config_symbol,
        location, &state->report_island));
    loom_builder_restore(builder, saved_ip);
    state->has_report_island = true;
  }
  *out_island = &state->report_island;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_get_trap_island(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_amdgpu_sanitizer_trap_island_t** out_island) {
  *out_island = NULL;
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  if (!state->has_trap_island) {
    loom_builder_ip_t saved_ip = loom_builder_save(builder);
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_trap_island(
        builder, descriptor_set, saved_ip.block, location,
        &state->trap_island));
    loom_builder_restore(builder, saved_ip);
    state->has_trap_island = true;
  }
  *out_island = &state->trap_island;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_vgpr_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_low_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
                                    vgpr_type, out_low_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_report_coordinates(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_report_coordinates_t* out_coordinates) {
  *out_coordinates = (loom_amdgpu_sanitizer_race_report_coordinates_t){0};
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);

  loom_value_id_t dispatch_ptr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_lookup_current_dispatch_ptr(context, &dispatch_ptr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, dispatch_ptr, 2, source_op->location,
      &out_coordinates->dispatch_ptr));

  for (uint32_t dimension = 0; dimension < LOOM_KERNEL_DIMENSION_COUNT_;
       ++dimension) {
    loom_value_id_t workgroup_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_current_workgroup_id(
        context, (loom_kernel_dimension_t)dimension, &workgroup_id));
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
        builder, descriptor_set, workgroup_id, 1, source_op->location,
        &out_coordinates->workgroup_ids[dimension]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_branch_disabled_entry(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_entry_interposition_t* entry,
    bool has_race_observation) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t zero_sgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, sgpr_type,
      &zero_sgpr));
  loom_value_id_t zero_vgpr64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_constant_u64(context, source_op,
                                                            0, &zero_vgpr64));
  loom_value_id_t disabled_args[6] = {
      zero_sgpr,
      zero_vgpr64,
      zero_sgpr,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  if (has_race_observation) {
    loom_value_id_t zero_vgpr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
        context, source_op, 0, &zero_vgpr));
    disabled_args[3] = zero_vgpr64;
    disabled_args[4] = zero_vgpr;
    disabled_args[5] = zero_vgpr;
  }
  return loom_amdgpu_sanitizer_race_branch_to_entry_body(
      context, entry, disabled_args, source_op->location);
}

static iree_status_t loom_amdgpu_sanitizer_race_state_vgpr_cmp_mask(
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

static iree_status_t loom_amdgpu_sanitizer_race_emit_capacity_failure(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_shadow_layout_t* layout,
    loom_value_id_t workitem_linear_id) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, 0, &zero));
  loom_value_id_t leader_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_state_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      workitem_linear_id, zero, &leader_mask));

  const loom_sanitizer_reporting_mode_t reporting_mode =
      loom_low_lower_context_sanitizer_reporting_mode(context);
  switch (reporting_mode) {
    case LOOM_SANITIZER_REPORTING_MODE_TRAP: {
      const loom_amdgpu_sanitizer_trap_island_t* island = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_trap_island(
          context, source_op->location, &island));
      loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_build_sanitizer_trap_failure_mask_branch_to_island(
              builder, descriptor_set, island, leader_mask, source_op->location,
              &branch));
      loom_op_t* return_op = NULL;
      return loom_low_return_build(builder, /*values=*/NULL,
                                   /*value_count=*/0, source_op->location,
                                   &return_op);
    }
    case LOOM_SANITIZER_REPORTING_MODE_DEFAULT:
    case LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY: {
      loom_amdgpu_feedback_failure_branch_t branch = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_failure_mask_split(
          builder, descriptor_set, leader_mask, source_op->location, &branch));

      const loom_amdgpu_sanitizer_race_report_coordinates_t* coordinates = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_report_coordinates(
          context, &coordinates));
      const loom_amdgpu_feedback_packet_source_t source = {
          .dispatch_ptr = coordinates->dispatch_ptr,
          .workgroup_id_x = coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_X],
          .workitem_id_x = zero,
      };
      loom_value_id_t shadow_cell_size = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
          context, source_op, LOOM_AMDGPU_TSAN_SHADOW_CELL_BYTE_LENGTH,
          &shadow_cell_size));
      loom_value_id_t shadow_history_count = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
          context, source_op, LOOM_AMDGPU_TSAN_SHADOW_HISTORY_COUNT,
          &shadow_history_count));
      const loom_amdgpu_sanitizer_race_shadow_capacity_report_t report = {
          .shadow_size = config->shadow_size,
          .required_shadow_end = layout->required_shadow_end,
          .workgroup_shadow_stride = layout->workgroup_shadow_stride,
          .workgroup_ordinal = layout->workgroup_ordinal,
          .workgroup_local_memory_size = layout->workgroup_local_memory_size,
          .shadow_cell_size = shadow_cell_size,
          .shadow_history_count = shadow_history_count,
          .current_workgroup_ids =
              {
                  coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_X],
                  coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_Y],
                  coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_Z],
              },
          .current_workitem_ids = {zero, zero, zero},
      };
      loom_symbol_ref_t feedback_config_symbol = loom_symbol_ref_null();
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_feedback_config_symbol(
          context, &feedback_config_symbol));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_build_sanitizer_race_shadow_capacity_report_terminate(
              builder, descriptor_set, feedback_config_symbol, &source, &report,
              source_op->location));
      loom_builder_set_block(builder, branch.continuation_block);
      loom_op_t* return_op = NULL;
      return loom_low_return_build(builder, /*values=*/NULL,
                                   /*value_count=*/0, source_op->location,
                                   &return_op);
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported AMDGPU sanitizer reporting mode");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_sanitizer_race_emit_entry_setup(
    loom_low_lower_context_t* context) {
  const loom_op_t* first_race_op = NULL;
  const loom_op_t* first_race_observation_op = NULL;
  const iree_host_size_t plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    if (selected_plan.plan.id == LOOM_OP_SANITIZER_RACE_ACCESS ||
        selected_plan.plan.id == LOOM_OP_SANITIZER_RACE_FRAGMENT_ACCESS) {
      if (first_race_observation_op == NULL) {
        first_race_observation_op = selected_plan.source_op;
      }
      if (first_race_op == NULL) {
        first_race_op = selected_plan.source_op;
      }
    } else if (selected_plan.plan.id == LOOM_OP_SANITIZER_RACE_SYNC &&
               first_race_op == NULL) {
      first_race_op = selected_plan.source_op;
    }
  }
  if (first_race_op == NULL) {
    return iree_ok_status();
  }

  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  if (state->has_config_values) {
    return iree_ok_status();
  }

  loom_symbol_ref_t config_symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_tsan_config_symbol(context, &config_symbol));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_config_values(
      context, first_race_op, config_symbol, &state->config_values));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_config_guard_values(
      context, first_race_op, state->config_values.flags,
      &state->required_config_flags, &state->active_config_flags));

  const loom_amdgpu_sanitizer_race_shadow_layout_config_t shadow_config = {
      .shadow_size = state->config_values.shadow_size,
  };
  loom_amdgpu_sanitizer_race_shadow_layout_t shadow_layout = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_race_shadow_layout(
      context, first_race_op, &shadow_config, &shadow_layout));
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_workitem_linear_id(
      context, first_race_op, vgpr_type, &state->workitem_linear_id));
  const loom_sanitizer_reporting_mode_t reporting_mode =
      loom_low_lower_context_sanitizer_reporting_mode(context);
  if (reporting_mode != LOOM_SANITIZER_REPORTING_MODE_TRAP) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_report_coordinates(
        context, first_race_op, &state->report_coordinates));
    state->has_report_coordinates = true;
  }

  state->has_config_values = true;
  state->has_topology_values = true;

  const bool has_race_observation = first_race_observation_op != NULL;
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  loom_type_t entry_state_arg_types[6] = {
      sgpr_type,        vgpr_x2_type,     sgpr_type,
      loom_type_none(), loom_type_none(), loom_type_none(),
  };
  uint16_t entry_state_arg_count = 3;
  if (has_race_observation) {
    entry_state_arg_types[entry_state_arg_count++] = vgpr_x2_type;
    entry_state_arg_types[entry_state_arg_count++] = vgpr_type;
    entry_state_arg_types[entry_state_arg_count++] = vgpr_type;
  }
  loom_low_lower_entry_interposition_t entry = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_entry_block(
      context, entry_state_arg_types, entry_state_arg_count, &entry));
  IREE_ASSERT_EQ(entry.target_arg_count, entry_state_arg_count);
  state->instrumentation_flags = entry.target_args[0];
  state->workgroup_shadow_record_offset = entry.target_args[1];
  state->workgroup_chunk_state_bitmap_size = entry.target_args[2];
  state->has_instrumentation_flags = true;
  state->has_workgroup_shadow_record_offset = true;
  state->has_workgroup_chunk_state_bitmap_size = true;
  if (has_race_observation) {
    state->workgroup_shadow_history_offset = entry.target_args[3];
    state->has_workgroup_shadow_history_offset = true;
    state->shadow_entry_base = (loom_amdgpu_sanitizer_race_shadow_entry_base_t){
        .low = entry.target_args[4],
        .high = entry.target_args[5],
    };
    state->has_shadow_entry_base = true;
  }

  loom_value_id_t enabled_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_required_config_scc(
      context, first_race_op, &enabled_scc));
  loom_block_t* config_disabled_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_split_entry_guard_blocks(
      loom_low_lower_context_builder(context), enabled_scc,
      first_race_op->location, /*out_enabled_block=*/NULL,
      &config_disabled_block));

  loom_block_t* capacity_invalid_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_split_entry_guard_blocks(
      loom_low_lower_context_builder(context), shadow_layout.in_bounds_scc,
      first_race_op->location, /*out_enabled_block=*/NULL,
      &capacity_invalid_block));

  loom_value_id_t enabled_args[6] = {
      state->required_config_flags, LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,        LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,        LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      loom_low_lower_context_builder(context),
      loom_low_lower_context_descriptor_set(context),
      shadow_layout.workgroup_shadow_offset, /*expected_unit_count=*/2,
      first_race_op->location, &enabled_args[1]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, first_race_op, shadow_layout.workgroup_chunk_state_bitmap_size,
      /*index=*/0, sgpr_type, &enabled_args[2]));
  loom_value_id_t dispatch_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_lookup_current_dispatch_id(context, &dispatch_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_establish_workgroup_lifetime(
      context, first_race_op, &state->config_values, &shadow_layout,
      state->workitem_linear_id, dispatch_id));
  if (has_race_observation) {
    loom_value_id_t history_data_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
        loom_low_lower_context_builder(context),
        loom_low_lower_context_descriptor_set(context),
        shadow_layout.workgroup_history_data_offset,
        /*expected_unit_count=*/2, first_race_op->location,
        &history_data_offset));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_vgpr64_add(context, first_race_op, enabled_args[1],
                                    history_data_offset, &enabled_args[3]));
    loom_value_id_t generation_low = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_sanitizer_race_emit_dispatch_generation_low(
            context, first_race_observation_op, dispatch_id, &generation_low));
    loom_amdgpu_sanitizer_race_shadow_entry_base_t enabled_base = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_shadow_entry_base(
        context, first_race_observation_op, state->workitem_linear_id,
        generation_low, &enabled_base));
    enabled_args[4] = enabled_base.low;
    enabled_args[5] = enabled_base.high;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_branch_to_entry_body(
      context, &entry, enabled_args, first_race_op->location));

  loom_builder_set_block(loom_low_lower_context_builder(context),
                         capacity_invalid_block);
  loom_block_t* later_invalid_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_split_entry_guard_blocks(
      loom_low_lower_context_builder(context), shadow_layout.first_invalid_scc,
      first_race_op->location, /*out_enabled_block=*/NULL,
      &later_invalid_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_emit_capacity_failure(
      context, first_race_op, &state->config_values, &shadow_layout,
      state->workitem_linear_id));

  loom_builder_set_block(loom_low_lower_context_builder(context),
                         later_invalid_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_branch_disabled_entry(
      context, first_race_op, &entry, has_race_observation));

  loom_builder_set_block(loom_low_lower_context_builder(context),
                         config_disabled_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_branch_disabled_entry(
      context, first_race_op, &entry, has_race_observation));
  loom_builder_set_block(loom_low_lower_context_builder(context),
                         entry.body_block);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_build_config_guard(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_config_values_t* out_config,
    loom_block_t** out_continuation_block) {
  *out_config = (loom_amdgpu_sanitizer_race_config_values_t){0};
  *out_continuation_block = NULL;

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_config_values(context, out_config));
  loom_value_id_t enabled_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_runtime_active_scc(
      context, source_op, &enabled_scc));
  loom_block_t* enabled_block = NULL;
  return loom_amdgpu_sanitizer_race_split_enabled_block(
      loom_low_lower_context_builder(context), enabled_scc, source_op->location,
      &enabled_block, out_continuation_block);
}
