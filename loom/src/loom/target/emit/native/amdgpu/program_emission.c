// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/program_emission.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/kernel_assembly.h"
#include "loom/target/emit/native/amdgpu/kernel_entry.h"

static iree_string_view_t loom_amdgpu_program_emission_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT_LT(symbol->name_id, module->strings.count);
  return loom_string_table_get(&module->strings, symbol->name_id);
}

static iree_status_t loom_amdgpu_program_emission_encode_function(
    const loom_amdgpu_function_compilation_t* compilation,
    loom_amdgpu_function_return_mode_t return_mode,
    loom_amdgpu_encode_instruction_stream_flags_t flags,
    loom_amdgpu_encoded_instruction_stream_t* out_stream,
    iree_arena_allocator_t* arena) {
  const loom_amdgpu_encode_instruction_stream_options_t options = {
      .packet_plan = &compilation->plan.packet_plan,
      .storage_layout = &compilation->plan.storage_layout,
      .flags = flags,
      .return_mode = return_mode,
  };
  return loom_amdgpu_encode_instruction_stream_result_with_options(
      &compilation->frame.schedule, &compilation->frame.allocation, &options,
      out_stream, arena);
}

iree_status_t loom_amdgpu_program_artifact_emission_build(
    const loom_amdgpu_program_artifact_t* artifact,
    loom_amdgpu_program_artifact_emission_t* out_emission,
    iree_arena_allocator_t* table_arena) {
  *out_emission = (loom_amdgpu_program_artifact_emission_t){
      .artifact = artifact,
  };
  if (artifact->helper_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_hsaco_function_t* functions = NULL;
  loom_amdgpu_encoded_instruction_stream_t* helper_streams = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(table_arena, artifact->helper_count,
                                sizeof(*functions), (void**)&functions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table_arena, artifact->helper_count, sizeof(*helper_streams),
      (void**)&helper_streams));
  memset(functions, 0, artifact->helper_count * sizeof(*functions));
  memset(helper_streams, 0, artifact->helper_count * sizeof(*helper_streams));

  for (iree_host_size_t i = 0; i < artifact->helper_count; ++i) {
    const loom_amdgpu_program_callable_t* callable = &artifact->helpers[i];
    IREE_ASSERT_EQ(callable->helper_ordinal, i);
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_emission_encode_function(
        &callable->compilation, LOOM_AMDGPU_FUNCTION_RETURN_MODE_CALLER,
        LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_NONE, &helper_streams[i],
        table_arena));
    IREE_ASSERT_EQ(helper_streams[i].internal_call_fixup_count, 0);
    functions[i] = (loom_amdgpu_hsaco_function_t){
        .name = loom_amdgpu_program_emission_symbol_name(artifact->module,
                                                         callable->function),
        .text = helper_streams[i].text,
        .text_fixups = helper_streams[i].text_fixups,
        .text_fixup_count = helper_streams[i].text_fixup_count,
    };
  }

  out_emission->functions = functions;
  out_emission->helper_streams = helper_streams;
  out_emission->function_count = artifact->helper_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_program_emission_build_call_fixups(
    const loom_amdgpu_program_artifact_root_t* root,
    const loom_amdgpu_encoded_instruction_stream_t* entry_stream,
    uint64_t entry_byte_offset, iree_arena_allocator_t* arena,
    const loom_amdgpu_hsaco_call_fixup_t** out_call_fixups) {
  *out_call_fixups = NULL;
  if (entry_stream->internal_call_fixup_count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_hsaco_call_fixup_t* call_fixups = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, entry_stream->internal_call_fixup_count,
                                sizeof(*call_fixups), (void**)&call_fixups));
  for (iree_host_size_t i = 0; i < entry_stream->internal_call_fixup_count;
       ++i) {
    const loom_amdgpu_internal_call_fixup_t* source =
        &entry_stream->internal_call_fixups[i];
    IREE_ASSERT(loom_symbol_ref_is_valid(source->callee));
    IREE_ASSERT_EQ(source->callee.module_id, 0u);
    IREE_ASSERT_LT(source->callee.symbol_id, root->call_table->symbol_count);
    const loom_amdgpu_program_callable_t* callable =
        root->call_table->callables_by_symbol[source->callee.symbol_id];
    IREE_ASSERT_NE(callable, NULL);
    loom_amdgpu_hsaco_call_fixup_t* target = &call_fixups[i];
    target->target_function_ordinal = callable->helper_ordinal;
    if (!iree_checked_add_u64(entry_byte_offset, source->base_pc_byte_offset,
                              &target->base_pc_byte_offset) ||
        !iree_checked_add_u64(entry_byte_offset,
                              source->low_literal_byte_offset,
                              &target->low_literal_byte_offset) ||
        !iree_checked_add_u64(entry_byte_offset,
                              source->high_literal_byte_offset,
                              &target->high_literal_byte_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU retained-program call fixup offset overflowed");
    }
  }
  *out_call_fixups = call_fixups;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_program_emission_append_listing(
    const loom_amdgpu_program_compilation_t* compilation,
    const loom_amdgpu_program_artifact_emission_t* artifact_emission,
    const loom_amdgpu_kernel_plan_t* entry_plan,
    const loom_amdgpu_encoded_instruction_stream_t* entry_stream,
    bool include_artifact_functions, iree_string_builder_t* target_listing,
    iree_arena_allocator_t* scratch_arena) {
  if (target_listing == NULL) {
    return iree_ok_status();
  }
  const iree_host_size_t helper_count =
      include_artifact_functions ? artifact_emission->function_count : 0;
  iree_host_size_t function_count = 0;
  if (!iree_host_size_checked_add(helper_count, 1, &function_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU assembly function count overflowed");
  }
  loom_amdgpu_kernel_program_assembly_function_t* functions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, function_count, sizeof(*functions), (void**)&functions));
  functions[0] = (loom_amdgpu_kernel_program_assembly_function_t){
      .schedule = &compilation->entry.frame.schedule,
      .allocation = &compilation->entry.frame.allocation,
      .fragment_options =
          {
              .packet_plan = &entry_plan->function.packet_plan,
              .storage_layout = &entry_plan->function.storage_layout,
              .instruction_layout = &entry_stream->layout,
              .return_mode = LOOM_AMDGPU_FUNCTION_RETURN_MODE_END_PROGRAM,
          },
  };
  const loom_amdgpu_program_artifact_t* artifact = artifact_emission->artifact;
  for (iree_host_size_t i = 0; i < helper_count; ++i) {
    const loom_amdgpu_program_callable_t* callable = &artifact->helpers[i];
    functions[i + 1] = (loom_amdgpu_kernel_program_assembly_function_t){
        .symbol = artifact_emission->functions[i].name,
        .schedule = &callable->compilation.frame.schedule,
        .allocation = &callable->compilation.frame.allocation,
        .fragment_options =
            {
                .packet_plan = &callable->compilation.plan.packet_plan,
                .storage_layout = &callable->compilation.plan.storage_layout,
                .instruction_layout =
                    &artifact_emission->helper_streams[i].layout,
                .return_mode = LOOM_AMDGPU_FUNCTION_RETURN_MODE_CALLER,
            },
    };
  }
  if (iree_string_builder_size(target_listing) != 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(target_listing, "\n\n"));
  }
  return loom_amdgpu_emit_kernel_program_assembly(
      &entry_plan->record, functions, function_count, target_listing,
      scratch_arena);
}

static iree_status_t loom_amdgpu_program_emission_build_internal(
    const loom_amdgpu_program_compilation_t* compilation,
    const loom_amdgpu_program_artifact_emission_t* artifact_emission,
    const loom_amdgpu_kernel_plan_t* entry_plan,
    loom_amdgpu_kernel_emission_flags_t flags,
    bool include_artifact_functions_in_listing,
    iree_string_builder_t* target_listing,
    loom_amdgpu_kernel_emission_result_t* out_result,
    iree_arena_allocator_t* table_arena,
    iree_arena_allocator_t* scratch_arena) {
  IREE_ASSERT(compilation->has_retained_calls);
  const loom_amdgpu_program_artifact_root_t* root = compilation->artifact_root;
  IREE_ASSERT_NE(root, NULL);
  IREE_ASSERT_GT(root->helper_count, 0);
  IREE_ASSERT_NE(root->call_table, NULL);
  IREE_ASSERT_EQ(artifact_emission->artifact->module,
                 compilation->entry.frame.module);
  const loom_amdgpu_kernel_record_t* record = &entry_plan->record;

  const loom_amdgpu_encode_instruction_stream_flags_t entry_flags =
      iree_any_bit_set(
          flags, LOOM_AMDGPU_KERNEL_EMISSION_FLAG_CAPTURE_NATIVE_INSERTIONS)
          ? LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_CAPTURE_NATIVE_INSERTIONS
          : LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_NONE;
  loom_amdgpu_encoded_instruction_stream_t entry_stream = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_emission_encode_function(
      &compilation->entry, LOOM_AMDGPU_FUNCTION_RETURN_MODE_END_PROGRAM,
      entry_flags, &entry_stream, table_arena));

  const loom_amdgpu_kernel_entry_envelope_t* entry_envelope =
      record->entry_envelope;
  iree_const_byte_span_t kernel_text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* kernel_text_fixups = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_entry_prepend_text(
      entry_envelope, entry_stream.text, entry_stream.text_fixups,
      entry_stream.text_fixup_count, &kernel_text, &kernel_text_fixups,
      table_arena));
  const loom_amdgpu_hsaco_call_fixup_t* call_fixups = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_emission_build_call_fixups(
      root, &entry_stream, entry_envelope->text.data_length, table_arena,
      &call_fixups));

  uint64_t body_instruction_count = entry_stream.instruction_count;
  for (iree_host_size_t i = 0; i < root->helper_count; ++i) {
    const loom_symbol_id_t helper_symbol_id = root->helper_symbols[i];
    const loom_amdgpu_program_callable_t* callable =
        root->call_table->callables_by_symbol[helper_symbol_id];
    IREE_ASSERT_NE(callable, NULL);
    IREE_ASSERT_LT(callable->helper_ordinal, artifact_emission->function_count);
    if (!iree_checked_add_u64(
            body_instruction_count,
            artifact_emission->helper_streams[callable->helper_ordinal]
                .instruction_count,
            &body_instruction_count)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "AMDGPU retained-program instruction count overflowed");
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_emission_append_listing(
      compilation, artifact_emission, entry_plan, &entry_stream,
      include_artifact_functions_in_listing, target_listing, scratch_arena));

  *out_result = (loom_amdgpu_kernel_emission_result_t){
      .kernel =
          {
              .metadata = record->metadata,
              .descriptor_options =
                  {
                      .flags = record->descriptor_flags,
                      .minimum_user_sgpr_count = record->user_sgpr_count,
                      .accum_offset = record->accum_offset,
                  },
              .text = kernel_text,
              .text_fixups = kernel_text_fixups,
              .text_fixup_count = entry_stream.text_fixup_count,
              .call_fixups = call_fixups,
              .call_fixup_count = entry_stream.internal_call_fixup_count,
          },
      .body_instruction_count = body_instruction_count,
      .entry_instruction_count = entry_envelope->instruction_count,
      .native_insertions = entry_stream.native_insertions,
      .native_insertion_count = entry_stream.native_insertion_count,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_program_emission_build(
    const loom_amdgpu_program_compilation_t* compilation,
    const loom_amdgpu_program_artifact_emission_t* artifact_emission,
    const loom_amdgpu_kernel_plan_t* entry_plan,
    loom_amdgpu_kernel_emission_flags_t flags,
    bool include_artifact_functions_in_listing,
    iree_string_builder_t* target_listing,
    loom_amdgpu_kernel_emission_result_t* out_result,
    iree_arena_allocator_t* table_arena) {
  *out_result = (loom_amdgpu_kernel_emission_result_t){0};
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(table_arena->block_pool, &scratch_arena);
  const iree_status_t status = loom_amdgpu_program_emission_build_internal(
      compilation, artifact_emission, entry_plan, flags,
      include_artifact_functions_in_listing, target_listing, out_result,
      table_arena, &scratch_arena);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
