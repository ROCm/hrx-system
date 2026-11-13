// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/artifact_symbols.h"

#include <string.h>

#include "iree/base/bitmap.h"
#include "loom/target/artifact_diagnostics.h"
#include "loom/target/emit/native/amdgpu/symbol_name.h"

static iree_status_t loom_amdgpu_artifact_symbol_claim(
    const loom_module_t* module, iree_string_view_t symbol,
    const loom_op_t* source_op, iree_diagnostic_emitter_t emitter,
    iree_bitmap_t claimed_names, bool* accepted) {
  if (symbol.size > UINT32_MAX - 1u) {
    *accepted = false;
    return loom_target_artifact_emit_limit(
        emitter, source_op, IREE_SV("AMDGPU HSACO"),
        IREE_SV("symbol name byte length"), symbol.size, UINT32_MAX - 1u);
  }
  if (!loom_amdgpu_symbol_name_is_valid(symbol)) {
    *accepted = false;
    return loom_target_artifact_emit_symbol_requirement(
        emitter, source_op, IREE_SV("AMDGPU HSACO"), symbol,
        IREE_SV("a symbol beginning with a letter, '_' or '$' and containing "
                "only letters, digits, '_', '$' or '.'"));
  }
  const loom_string_id_t name_id = loom_module_lookup_string(module, symbol);
  if (name_id == LOOM_STRING_ID_INVALID) {
    // Authored names are interned. Descriptor and runtime names need not be;
    // their producers own uniqueness within disjoint generated namespaces.
    // Any collision with authored input therefore has an interned name.
    return iree_ok_status();
  }
  IREE_ASSERT_LT(name_id, claimed_names.bit_count);
  if (iree_bitmap_test(claimed_names, name_id)) {
    *accepted = false;
    return loom_target_artifact_emit_symbol_requirement(
        emitter, source_op, IREE_SV("AMDGPU HSACO"), symbol,
        IREE_SV(
            "a name unique across kernels, descriptors, functions, and data"));
  }
  iree_bitmap_set(claimed_names, name_id);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_artifact_symbols_validate(
    const loom_module_t* module, const loom_amdgpu_hsaco_input_t* input,
    loom_target_entry_list_t entries,
    const loom_amdgpu_program_artifact_t* artifact,
    const loom_amdgpu_data_symbol_directory_t* data_symbols,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    bool* out_accepted) {
  *out_accepted = false;
  const iree_host_size_t word_count =
      iree_bitmap_calculate_words(module->strings.count);
  uint64_t* claimed_words = NULL;
  if (word_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, word_count, sizeof(*claimed_words), (void**)&claimed_words));
    memset(claimed_words, 0, word_count * sizeof(*claimed_words));
  }
  const iree_bitmap_t claimed_names = {
      .bit_count = module->strings.count,
      .words = claimed_words,
  };

  bool accepted = true;
  // Claim runtime/data names first: a conflicting authored kernel or helper
  // then owns the diagnostic even when the runtime name has no declaration.
  for (iree_host_size_t i = 0; i < input->data_symbol_count && accepted; ++i) {
    const loom_symbol_id_t source_symbol = data_symbols->source_symbols[i];
    const loom_op_t* source_op =
        source_symbol != LOOM_SYMBOL_ID_INVALID
            ? module->symbols.entries[source_symbol].defining_op
            : NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_symbol_claim(
        module, input->data_symbols[i].name, source_op, emitter, claimed_names,
        &accepted));
  }
  for (iree_host_size_t i = 0; i < input->kernel_count && accepted; ++i) {
    const loom_amdgpu_metadata_kernel_t* kernel = &input->kernels[i].metadata;
    const loom_target_entry_t* entry = &entries.values[i];
    loom_diagnostic_export_scope_t scope = {
        .export_name = loom_target_facts_export_name(entry->target_facts),
        .target = emitter,
    };
    const iree_diagnostic_emitter_t scoped_emitter =
        loom_diagnostic_export_scope_emitter(&scope);
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_symbol_claim(
        module, kernel->name, entry->func.op, scoped_emitter, claimed_names,
        &accepted));
    if (accepted) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_symbol_claim(
          module, kernel->descriptor_symbol, entry->func.op, scoped_emitter,
          claimed_names, &accepted));
    }
  }
  for (iree_host_size_t i = 0; i < input->function_count && accepted; ++i) {
    const loom_amdgpu_program_callable_t* helper = &artifact->helpers[i];
    loom_diagnostic_export_scope_t scope = {
        .export_name =
            loom_target_facts_export_name(helper->target.target_facts),
        .target = emitter,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_symbol_claim(
        module, input->functions[i].name, helper->compilation.frame.function_op,
        loom_diagnostic_export_scope_emitter(&scope), claimed_names,
        &accepted));
  }
  *out_accepted = accepted;
  return iree_ok_status();
}
