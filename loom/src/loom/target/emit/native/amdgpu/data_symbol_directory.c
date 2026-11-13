// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/data_symbol_directory.h"

#include <string.h>

#include "loom/ops/global/ops.h"
#include "loom/target/artifact_diagnostics.h"

static iree_string_view_t loom_amdgpu_data_symbol_directory_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  IREE_ASSERT(symbol->name_id != LOOM_STRING_ID_INVALID &&
              symbol->name_id < module->strings.count);
  return loom_string_table_get(&module->strings, symbol->name_id);
}

static bool loom_amdgpu_data_symbol_directory_has_runtime_global(
    const loom_module_t* module, iree_string_view_t name) {
  const loom_string_id_t name_id = loom_module_lookup_string(module, name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return false;
  }
  const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return false;
  }
  const loom_op_t* defining_op = module->symbols.entries[symbol_id].defining_op;
  return defining_op == NULL || loom_global_rodata_decl_isa(defining_op);
}

static loom_amdgpu_runtime_global_flags_t
loom_amdgpu_data_symbol_directory_runtime_globals(const loom_module_t* module) {
  loom_amdgpu_runtime_global_flags_t flags = LOOM_AMDGPU_RUNTIME_GLOBAL_NONE;
  if (loom_amdgpu_data_symbol_directory_has_runtime_global(
          module, IREE_SV(LOOM_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME))) {
    flags |= LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG;
  }
  if (loom_amdgpu_data_symbol_directory_has_runtime_global(
          module, IREE_SV(LOOM_AMDGPU_ASAN_CONFIG_GLOBAL_NAME))) {
    flags |= LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG;
  }
  if (loom_amdgpu_data_symbol_directory_has_runtime_global(
          module, IREE_SV(LOOM_AMDGPU_TSAN_CONFIG_GLOBAL_NAME))) {
    flags |= LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG;
  }
  return flags;
}

static void loom_amdgpu_data_symbol_directory_index_runtime_globals(
    const loom_module_t* module,
    const loom_amdgpu_hsaco_data_symbol_t* runtime_symbols,
    iree_host_size_t runtime_symbol_count, uint32_t* ordinals_by_module_symbol,
    loom_symbol_id_t* source_symbols) {
  for (iree_host_size_t i = 0; i < runtime_symbol_count; ++i) {
    const loom_string_id_t name_id =
        loom_module_lookup_string(module, runtime_symbols[i].name);
    if (name_id == LOOM_STRING_ID_INVALID) {
      continue;
    }
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    IREE_ASSERT_LT(symbol_id, module->symbols.count);
    IREE_ASSERT_EQ(ordinals_by_module_symbol[symbol_id],
                   LOOM_AMDGPU_DATA_SYMBOL_ORDINAL_INVALID);
    ordinals_by_module_symbol[symbol_id] = (uint32_t)i;
    source_symbols[i] = symbol_id;
  }
}

iree_status_t loom_amdgpu_data_symbol_directory_build(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_amdgpu_data_symbol_directory_t* out_directory) {
  *out_directory = (loom_amdgpu_data_symbol_directory_t){0};
  const loom_amdgpu_runtime_global_flags_t runtime_globals =
      loom_amdgpu_data_symbol_directory_runtime_globals(module);

  iree_host_size_t authored_symbol_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_op_t* defining_op = module->symbols.entries[i].defining_op;
    if (defining_op && loom_global_rodata_def_isa(defining_op)) {
      ++authored_symbol_count;
    }
  }
  const iree_host_size_t runtime_symbol_count =
      loom_amdgpu_runtime_global_count(runtime_globals);
  const iree_host_size_t symbol_count =
      runtime_symbol_count + authored_symbol_count;

  uint32_t* ordinals_by_module_symbol = NULL;
  if (module->symbols.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->symbols.count, sizeof(*ordinals_by_module_symbol),
        (void**)&ordinals_by_module_symbol));
    memset(ordinals_by_module_symbol, 0xFF,
           module->symbols.count * sizeof(*ordinals_by_module_symbol));
  }

  loom_amdgpu_hsaco_data_symbol_t* symbols = NULL;
  loom_symbol_id_t* source_symbols = NULL;
  if (symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, symbol_count, sizeof(*symbols), (void**)&symbols));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, symbol_count, sizeof(*source_symbols), (void**)&source_symbols));
    memset(source_symbols, 0xFF, symbol_count * sizeof(*source_symbols));
  }
  iree_host_size_t next_ordinal = 0;
  loom_amdgpu_runtime_global_symbols(runtime_globals, symbols, &next_ordinal);
  IREE_ASSERT_EQ(next_ordinal, runtime_symbol_count);
  loom_amdgpu_data_symbol_directory_index_runtime_globals(
      module, symbols, runtime_symbol_count, ordinals_by_module_symbol,
      source_symbols);

  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    const loom_op_t* op = symbol->defining_op;
    if (!op || !loom_global_rodata_def_isa(op)) {
      continue;
    }
    IREE_ASSERT_EQ(ordinals_by_module_symbol[i],
                   LOOM_AMDGPU_DATA_SYMBOL_ORDINAL_INVALID);
    ordinals_by_module_symbol[i] = (uint32_t)next_ordinal;
    const iree_const_byte_span_t contents = loom_global_rodata_def_contents(op);
    source_symbols[next_ordinal] = (loom_symbol_id_t)i;
    symbols[next_ordinal++] = (loom_amdgpu_hsaco_data_symbol_t){
        .name = loom_amdgpu_data_symbol_directory_symbol_name(module, symbol),
        .initial_contents = contents,
        .byte_length = contents.data_length,
        .alignment = loom_global_rodata_def_has_alignment(op)
                         ? (uint64_t)loom_global_rodata_def_alignment(op)
                         : 0,
    };
  }
  IREE_ASSERT_EQ(next_ordinal, symbol_count);

  *out_directory = (loom_amdgpu_data_symbol_directory_t){
      .module = module,
      .symbols = symbols,
      .source_symbols = source_symbols,
      .symbol_count = symbol_count,
      .ordinals_by_module_symbol = ordinals_by_module_symbol,
      .layout = iree_any_bit_set(runtime_globals,
                                 LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG)
                    ? LOOM_AMDGPU_HSACO_DATA_LAYOUT_ASAN_GLOBALS_V0
                    : LOOM_AMDGPU_HSACO_DATA_LAYOUT_PACKED,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_data_symbol_directory_bind_fixups(
    const loom_amdgpu_data_symbol_directory_t* directory,
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_data_symbol_fixup_t* fixups, iree_host_size_t fixup_count,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter,
    bool* out_accepted, const loom_amdgpu_hsaco_text_fixup_t** out_fixups) {
  *out_accepted = false;
  *out_fixups = NULL;
  if (fixup_count == 0) {
    *out_accepted = true;
    return iree_ok_status();
  }

  loom_amdgpu_hsaco_text_fixup_t* bound_fixups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, fixup_count, sizeof(*bound_fixups), (void**)&bound_fixups));
  for (iree_host_size_t i = 0; i < fixup_count; ++i) {
    const loom_symbol_id_t symbol_id = fixups[i].target_symbol_id;
    const uint32_t symbol_ordinal =
        directory->ordinals_by_module_symbol[symbol_id];
    if (symbol_ordinal == LOOM_AMDGPU_DATA_SYMBOL_ORDINAL_INVALID) {
      const loom_symbol_t* symbol =
          &directory->module->symbols.entries[symbol_id];
      const iree_string_view_t symbol_name =
          loom_string_table_get(&directory->module->strings, symbol->name_id);
      return loom_target_artifact_emit_symbol_requirement(
          emitter, schedule->nodes[fixups[i].node_index].op,
          IREE_SV("AMDGPU HSACO"), symbol_name,
          IREE_SV("a data definition in this artifact"));
    }
    const loom_amdgpu_hsaco_data_symbol_t* symbol =
        &directory->symbols[symbol_ordinal];
    if (fixups[i].target_symbol_byte_offset > symbol->byte_length) {
      return loom_target_artifact_emit_symbol_offset(
          emitter, schedule->nodes[fixups[i].node_index].op,
          IREE_SV("AMDGPU HSACO"), symbol->name,
          fixups[i].target_symbol_byte_offset, symbol->byte_length);
    }
    bound_fixups[i] = (loom_amdgpu_hsaco_text_fixup_t){
        .kind = fixups[i].kind,
        .target_data_symbol_ordinal = symbol_ordinal,
        .literal_byte_offset = fixups[i].literal_byte_offset,
        .base_pc_byte_offset = fixups[i].base_pc_byte_offset,
        .target_symbol_byte_offset = fixups[i].target_symbol_byte_offset,
    };
  }
  *out_fixups = bound_fixups;
  *out_accepted = true;
  return iree_ok_status();
}
