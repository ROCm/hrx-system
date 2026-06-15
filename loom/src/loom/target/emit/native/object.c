// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/object.h"

#include <inttypes.h>

static iree_status_t loom_native_object_validate_symbol_name(
    iree_string_view_t name, iree_host_size_t index) {
  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native object symbol %" PRIhsz " name is required",
                            index);
  }
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    if (name.data[i] == '\0') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "native object symbol %" PRIhsz
                              " name contains an embedded NUL",
                              index);
    }
  }
  return iree_ok_status();
}

static bool loom_native_object_checked_add_uint64(uint64_t lhs, uint64_t rhs,
                                                  uint64_t* out_result) {
  *out_result = lhs + rhs;
  return *out_result >= lhs;
}

static iree_status_t loom_native_object_validate_symbol(
    const loom_native_object_symbol_t* symbol, iree_host_size_t index,
    iree_host_size_t section_layout_count) {
  IREE_RETURN_IF_ERROR(
      loom_native_object_validate_symbol_name(symbol->name, index));
  switch (symbol->binding) {
    case LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL:
    case LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL:
    case LOOM_NATIVE_OBJECT_SYMBOL_BINDING_WEAK:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "native object symbol %" PRIhsz " binding is invalid", index);
  }
  switch (symbol->visibility) {
    case LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT:
    case LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_INTERNAL:
    case LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_HIDDEN:
    case LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_PROTECTED:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "native object symbol %" PRIhsz " visibility is invalid", index);
  }
  switch (symbol->kind) {
    case LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION:
    case LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "native object symbol %" PRIhsz " kind is invalid", index);
  }
  if (symbol->section_contribution_index >= section_layout_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "native object symbol %" PRIhsz
                            " section contribution index %" PRIhsz
                            " is outside the assembled contribution layout",
                            index, symbol->section_contribution_index);
  }
  return iree_ok_status();
}

iree_status_t loom_native_object_resolve_symbol_layouts(
    const loom_native_object_symbol_t* symbols, iree_host_size_t symbol_count,
    const loom_native_section_contribution_layout_t* section_layouts,
    iree_host_size_t section_layout_count,
    loom_native_object_symbol_layout_t* out_symbol_layouts) {
  if (symbol_count == 0) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    const loom_native_object_symbol_t* symbol = &symbols[i];
    IREE_RETURN_IF_ERROR(
        loom_native_object_validate_symbol(symbol, i, section_layout_count));
    const loom_native_section_contribution_layout_t* section_layout =
        &section_layouts[symbol->section_contribution_index];
    uint64_t final_offset = 0;
    if (!loom_native_object_checked_add_uint64(section_layout->section_offset,
                                               symbol->section_offset,
                                               &final_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "native object symbol %" PRIhsz " section offset overflows", i);
    }
    out_symbol_layouts[i] = (loom_native_object_symbol_layout_t){
        .section_index = section_layout->section_index,
        .section_offset = final_offset,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_native_object_validate_fixup(
    const loom_native_object_fixup_t* fixup, iree_host_size_t index,
    iree_host_size_t symbol_count, iree_host_size_t section_layout_count) {
  if (fixup->relocation_kind == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native object fixup %" PRIhsz " relocation kind is required", index);
  }
  if (fixup->target_symbol_index >= symbol_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "native object fixup %" PRIhsz
                            " target symbol index %" PRIhsz
                            " is outside the object symbol table",
                            index, fixup->target_symbol_index);
  }
  if (fixup->section_contribution_index >= section_layout_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "native object fixup %" PRIhsz
                            " section contribution index %" PRIhsz
                            " is outside the assembled contribution layout",
                            index, fixup->section_contribution_index);
  }
  return iree_ok_status();
}

iree_status_t loom_native_object_resolve_fixup_layouts(
    const loom_native_object_fixup_t* fixups, iree_host_size_t fixup_count,
    iree_host_size_t symbol_count,
    const loom_native_section_contribution_layout_t* section_layouts,
    iree_host_size_t section_layout_count,
    loom_native_object_fixup_layout_t* out_fixup_layouts) {
  if (fixup_count == 0) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < fixup_count; ++i) {
    const loom_native_object_fixup_t* fixup = &fixups[i];
    IREE_RETURN_IF_ERROR(loom_native_object_validate_fixup(
        fixup, i, symbol_count, section_layout_count));
    const loom_native_section_contribution_layout_t* section_layout =
        &section_layouts[fixup->section_contribution_index];
    uint64_t final_offset = 0;
    if (!loom_native_object_checked_add_uint64(section_layout->section_offset,
                                               fixup->section_offset,
                                               &final_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "native object fixup %" PRIhsz " section offset overflows", i);
    }
    out_fixup_layouts[i] = (loom_native_object_fixup_layout_t){
        .section_index = section_layout->section_index,
        .section_offset = final_offset,
    };
  }
  return iree_ok_status();
}
