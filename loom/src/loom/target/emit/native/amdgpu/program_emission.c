// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/program_emission.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "loom/ir/module.h"
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/kernel_assembly.h"

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

static iree_status_t loom_amdgpu_program_emission_copy_text_fixups(
    const loom_amdgpu_hsaco_text_fixup_t* source, iree_host_size_t source_count,
    uint64_t function_byte_offset, loom_amdgpu_hsaco_text_fixup_t* target,
    iree_host_size_t* inout_target_count) {
  for (iree_host_size_t i = 0; i < source_count; ++i) {
    loom_amdgpu_hsaco_text_fixup_t fixup = source[i];
    if (!iree_checked_add_u64(fixup.literal_byte_offset, function_byte_offset,
                              &fixup.literal_byte_offset) ||
        !iree_checked_add_u64(fixup.base_pc_byte_offset, function_byte_offset,
                              &fixup.base_pc_byte_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU retained-program text fixup offset overflowed");
    }
    target[(*inout_target_count)++] = fixup;
  }
  return iree_ok_status();
}

static void loom_amdgpu_program_emission_patch_internal_calls(
    const loom_amdgpu_program_compilation_t* compilation,
    const loom_amdgpu_encoded_instruction_stream_t* entry_stream,
    iree_host_size_t entry_byte_offset,
    const iree_host_size_t* helper_byte_offsets, uint8_t* text,
    iree_host_size_t text_length) {
  const loom_amdgpu_program_call_table_t* call_table = &compilation->call_table;
  for (iree_host_size_t i = 0; i < entry_stream->internal_call_fixup_count;
       ++i) {
    const loom_amdgpu_internal_call_fixup_t* fixup =
        &entry_stream->internal_call_fixups[i];
    IREE_ASSERT(loom_symbol_ref_is_valid(fixup->callee));
    IREE_ASSERT_EQ(fixup->callee.module_id, 0);
    IREE_ASSERT_LT(fixup->callee.symbol_id, call_table->symbol_count);
    const loom_amdgpu_program_callable_t* callable =
        call_table->callables_by_symbol[fixup->callee.symbol_id];
    IREE_ASSERT_NE(callable, NULL);
    IREE_ASSERT_LT(callable->helper_ordinal, compilation->helper_count);

    uint64_t base_pc_byte_offset = 0;
    uint64_t low_literal_byte_offset = 0;
    uint64_t high_literal_byte_offset = 0;
    const bool has_base_pc_byte_offset = iree_checked_add_u64(
        entry_byte_offset, fixup->base_pc_byte_offset, &base_pc_byte_offset);
    const bool has_low_literal_byte_offset =
        iree_checked_add_u64(entry_byte_offset, fixup->low_literal_byte_offset,
                             &low_literal_byte_offset);
    const bool has_high_literal_byte_offset =
        iree_checked_add_u64(entry_byte_offset, fixup->high_literal_byte_offset,
                             &high_literal_byte_offset);
    IREE_ASSERT(has_base_pc_byte_offset);
    IREE_ASSERT(has_low_literal_byte_offset);
    IREE_ASSERT(has_high_literal_byte_offset);
    (void)has_base_pc_byte_offset;
    (void)has_low_literal_byte_offset;
    (void)has_high_literal_byte_offset;
    const uint64_t target_byte_offset =
        helper_byte_offsets[callable->helper_ordinal];
    IREE_ASSERT_GE(target_byte_offset, base_pc_byte_offset);
    const uint64_t displacement = target_byte_offset - base_pc_byte_offset;
    IREE_ASSERT_GE(text_length, sizeof(uint32_t));
    IREE_ASSERT_LE(low_literal_byte_offset, text_length - sizeof(uint32_t));
    IREE_ASSERT_LE(high_literal_byte_offset, text_length - sizeof(uint32_t));
    iree_unaligned_store_le_u32(text + low_literal_byte_offset,
                                (uint32_t)displacement);
    iree_unaligned_store_le_u32(text + high_literal_byte_offset,
                                (uint32_t)(displacement >> 32));
  }
}

static iree_status_t loom_amdgpu_program_emission_build_internal(
    const loom_amdgpu_program_compilation_t* compilation,
    const loom_amdgpu_kernel_plan_t* entry_plan,
    loom_amdgpu_kernel_emission_flags_t flags,
    iree_string_builder_t* target_listing,
    loom_amdgpu_kernel_emission_result_t* out_result,
    iree_arena_allocator_t* table_arena,
    iree_arena_allocator_t* scratch_arena) {
  IREE_ASSERT(compilation->has_retained_calls);
  IREE_ASSERT_GT(compilation->helper_count, 0);
  const loom_amdgpu_kernel_record_t* record = &entry_plan->record;
  const loom_amdgpu_kernel_entry_envelope_t* entry_envelope =
      record->entry_envelope;

  const loom_amdgpu_encode_instruction_stream_flags_t entry_flags =
      iree_any_bit_set(
          flags, LOOM_AMDGPU_KERNEL_EMISSION_FLAG_CAPTURE_NATIVE_INSERTIONS)
          ? LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_CAPTURE_NATIVE_INSERTIONS
          : LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_NONE;
  loom_amdgpu_encoded_instruction_stream_t entry_stream = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_emission_encode_function(
      &compilation->entry, LOOM_AMDGPU_FUNCTION_RETURN_MODE_END_PROGRAM,
      entry_flags, &entry_stream, table_arena));

  loom_amdgpu_encoded_instruction_stream_t* helper_streams = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, compilation->helper_count, sizeof(*helper_streams),
      (void**)&helper_streams));
  memset(helper_streams, 0,
         compilation->helper_count * sizeof(*helper_streams));
  iree_host_size_t* helper_byte_offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, compilation->helper_count, sizeof(*helper_byte_offsets),
      (void**)&helper_byte_offsets));

  iree_host_size_t text_length = 0;
  if (!iree_host_size_checked_add(entry_envelope->text.data_length,
                                  entry_stream.text.data_length,
                                  &text_length)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU retained-program entry text length overflowed");
  }
  iree_host_size_t text_fixup_count = entry_stream.text_fixup_count;
  uint64_t body_instruction_count = entry_stream.instruction_count;
  for (iree_host_size_t i = 0; i < compilation->helper_count; ++i) {
    const loom_amdgpu_program_callable_t* callable = &compilation->helpers[i];
    IREE_ASSERT_EQ(callable->helper_ordinal, i);
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_emission_encode_function(
        &callable->compilation, LOOM_AMDGPU_FUNCTION_RETURN_MODE_CALLER,
        LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_NONE, &helper_streams[i],
        scratch_arena));
    IREE_ASSERT_EQ(helper_streams[i].internal_call_fixup_count, 0);
    helper_byte_offsets[i] = text_length;
    if (!iree_host_size_checked_add(
            text_length, helper_streams[i].text.data_length, &text_length) ||
        !iree_host_size_checked_add(text_fixup_count,
                                    helper_streams[i].text_fixup_count,
                                    &text_fixup_count) ||
        !iree_checked_add_u64(body_instruction_count,
                              helper_streams[i].instruction_count,
                              &body_instruction_count)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "AMDGPU retained-program text accounting overflowed");
    }
  }

  uint8_t* text = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(table_arena, text_length, (void**)&text));
  iree_host_size_t cursor = 0;
  if (entry_envelope->text.data_length != 0) {
    memcpy(text + cursor, entry_envelope->text.data,
           entry_envelope->text.data_length);
  }
  cursor += entry_envelope->text.data_length;
  const iree_host_size_t entry_byte_offset = cursor;
  if (entry_stream.text.data_length != 0) {
    memcpy(text + cursor, entry_stream.text.data,
           entry_stream.text.data_length);
  }
  cursor += entry_stream.text.data_length;
  for (iree_host_size_t i = 0; i < compilation->helper_count; ++i) {
    IREE_ASSERT_EQ(cursor, helper_byte_offsets[i]);
    if (helper_streams[i].text.data_length != 0) {
      memcpy(text + cursor, helper_streams[i].text.data,
             helper_streams[i].text.data_length);
    }
    cursor += helper_streams[i].text.data_length;
  }
  IREE_ASSERT_EQ(cursor, text_length);

  loom_amdgpu_hsaco_text_fixup_t* text_fixups = NULL;
  if (text_fixup_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(table_arena, text_fixup_count,
                                  sizeof(*text_fixups), (void**)&text_fixups));
  }
  iree_host_size_t copied_text_fixup_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_emission_copy_text_fixups(
      entry_stream.text_fixups, entry_stream.text_fixup_count,
      entry_byte_offset, text_fixups, &copied_text_fixup_count));
  for (iree_host_size_t i = 0; i < compilation->helper_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_emission_copy_text_fixups(
        helper_streams[i].text_fixups, helper_streams[i].text_fixup_count,
        helper_byte_offsets[i], text_fixups, &copied_text_fixup_count));
  }
  IREE_ASSERT_EQ(copied_text_fixup_count, text_fixup_count);
  loom_amdgpu_program_emission_patch_internal_calls(
      compilation, &entry_stream, entry_byte_offset, helper_byte_offsets, text,
      text_length);

  if (target_listing != NULL) {
    iree_host_size_t function_count = 0;
    const bool has_function_count = iree_host_size_checked_add(
        compilation->helper_count, 1, &function_count);
    IREE_ASSERT(has_function_count);
    (void)has_function_count;
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
                .instruction_layout = &entry_stream.layout,
                .return_mode = LOOM_AMDGPU_FUNCTION_RETURN_MODE_END_PROGRAM,
            },
    };
    const loom_module_t* module = compilation->entry.frame.module;
    for (iree_host_size_t i = 0; i < compilation->helper_count; ++i) {
      const loom_amdgpu_program_callable_t* callable = &compilation->helpers[i];
      functions[i + 1] = (loom_amdgpu_kernel_program_assembly_function_t){
          .symbol = loom_amdgpu_program_emission_symbol_name(
              module, callable->function),
          .schedule = &callable->compilation.frame.schedule,
          .allocation = &callable->compilation.frame.allocation,
          .fragment_options =
              {
                  .packet_plan = &callable->compilation.plan.packet_plan,
                  .storage_layout = &callable->compilation.plan.storage_layout,
                  .instruction_layout = &helper_streams[i].layout,
                  .return_mode = LOOM_AMDGPU_FUNCTION_RETURN_MODE_CALLER,
              },
      };
    }
    if (iree_string_builder_size(target_listing) != 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(target_listing, "\n\n"));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_kernel_program_assembly(
        record, functions, function_count, target_listing, scratch_arena));
  }

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
              .text = iree_make_const_byte_span(text, text_length),
              .text_fixups = text_fixups,
              .text_fixup_count = text_fixup_count,
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
    const loom_amdgpu_kernel_plan_t* entry_plan,
    loom_amdgpu_kernel_emission_flags_t flags,
    iree_string_builder_t* target_listing,
    loom_amdgpu_kernel_emission_result_t* out_result,
    iree_arena_allocator_t* table_arena) {
  *out_result = (loom_amdgpu_kernel_emission_result_t){0};
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(table_arena->block_pool, &scratch_arena);
  const iree_status_t status = loom_amdgpu_program_emission_build_internal(
      compilation, entry_plan, flags, target_listing, out_result, table_arena,
      &scratch_arena);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
