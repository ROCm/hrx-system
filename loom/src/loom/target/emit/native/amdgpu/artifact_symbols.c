// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/artifact_symbols.h"

#include <string.h>

#include "iree/base/bitmap.h"
#include "loom/target/emit/native/amdgpu/symbol_name.h"

static iree_status_t loom_amdgpu_artifact_symbol_claim(
    const loom_module_t* module, iree_string_view_t symbol,
    bool allow_synthesized_name, iree_bitmap_t claimed_names) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_symbol_name_validate(symbol));
  const loom_string_id_t name_id = loom_module_lookup_string(module, symbol);
  if (name_id == LOOM_STRING_ID_INVALID) {
    // Every non-synthesized name is retained from a module symbol or target
    // attribute. Descriptor and fixed runtime names need not already occur in
    // the module; their producers own uniqueness within disjoint namespaces.
    // A collision with any authored name makes the string interned and reaches
    // the bitmap path below.
    if (!allow_synthesized_name) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU artifact symbol '%.*s' is not retained by the module",
          (int)symbol.size, symbol.data);
    }
    return iree_ok_status();
  }
  IREE_ASSERT_LT(name_id, claimed_names.bit_count);
  if (iree_bitmap_test(claimed_names, name_id)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU artifact symbol '%.*s' is emitted more "
                            "than once",
                            (int)symbol.size, symbol.data);
  }
  iree_bitmap_set(claimed_names, name_id);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_artifact_symbols_validate(
    const loom_module_t* module, const loom_amdgpu_hsaco_input_t* input,
    iree_arena_allocator_t* arena) {
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

  for (iree_host_size_t i = 0; i < input->kernel_count; ++i) {
    const loom_amdgpu_metadata_kernel_t* kernel = &input->kernels[i].metadata;
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_symbol_claim(
        module, kernel->name, /*allow_synthesized_name=*/false, claimed_names));
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_symbol_claim(
        module, kernel->descriptor_symbol, /*allow_synthesized_name=*/true,
        claimed_names));
  }
  for (iree_host_size_t i = 0; i < input->function_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_symbol_claim(
        module, input->functions[i].name, /*allow_synthesized_name=*/false,
        claimed_names));
  }
  for (iree_host_size_t i = 0; i < input->data_symbol_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_symbol_claim(
        module, input->data_symbols[i].name, /*allow_synthesized_name=*/true,
        claimed_names));
  }
  return iree_ok_status();
}
