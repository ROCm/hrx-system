// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/reference_binding.h"

static bool loom_bytecode_reference_binding_mismatch(
    loom_bytecode_reader_decoder_t* decoder, uint64_t bytecode_offset) {
  return loom_bytecode_reader_emit_invalid_field(
      decoder, IREE_SV("TYPES"), IREE_SV("reference"), 0,
      IREE_SV("declaration"), bytecode_offset,
      IREE_SV("managed_type_key_disagrees_with_its_declaration"));
}

bool loom_bytecode_reference_binding_expect(
    loom_bytecode_reader_decoder_t* decoder, iree_arena_allocator_t* arena,
    loom_bytecode_reference_bindings_t* bindings,
    loom_string_id_t declaration_name_id, loom_string_id_t namespace_id,
    loom_string_id_t name_id, uint64_t bytecode_offset) {
  uint16_t index =
      loom_symbol_map_find(&bindings->pending, declaration_name_id);
  if (index != LOOM_SYMBOL_ID_INVALID) {
    const loom_bytecode_reference_binding_t* expected =
        &bindings->values[index];
    if (expected->namespace_id != namespace_id ||
        expected->name_id != name_id) {
      return loom_bytecode_reference_binding_mismatch(decoder, bytecode_offset);
    }
    return true;
  }
  if (bindings->count == LOOM_SYMBOL_ID_INVALID) {
    return loom_bytecode_reader_emit_count_exceeds(
        decoder, IREE_SV("managed_type_declarations"), bindings->count + 1,
        LOOM_SYMBOL_ID_INVALID, bytecode_offset);
  }
  LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_reader_check_status(
      decoder,
      iree_arena_grow_array(arena, bindings->count, bindings->count + 1,
                            sizeof(*bindings->values), &bindings->capacity,
                            (void**)&bindings->values)));
  LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_reader_check_status(
      decoder,
      loom_symbol_map_insert(&bindings->pending, arena, declaration_name_id,
                             (uint16_t)bindings->count)));
  bindings->values[bindings->count++] = (loom_bytecode_reference_binding_t){
      .bytecode_offset = bytecode_offset,
      .namespace_id = namespace_id,
      .name_id = name_id,
  };
  return true;
}

bool loom_bytecode_reference_binding_match(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reference_bindings_t* bindings,
    loom_string_id_t declaration_name_id, loom_string_id_t namespace_id,
    loom_string_id_t name_id) {
  const uint16_t index =
      loom_symbol_map_find(&bindings->pending, declaration_name_id);
  if (index == LOOM_SYMBOL_ID_INVALID) {
    return true;
  }
  const loom_bytecode_reference_binding_t* expected = &bindings->values[index];
  if (expected->namespace_id != namespace_id || expected->name_id != name_id) {
    return loom_bytecode_reference_binding_mismatch(decoder,
                                                    expected->bytecode_offset);
  }
  loom_symbol_map_erase(&bindings->pending, declaration_name_id);
  return true;
}

bool loom_bytecode_reference_bindings_finish(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reference_bindings_t* bindings) {
  if (!bindings->pending.count) {
    return true;
  }
  // Only the error path visits the index to locate an unresolved use.
  const loom_symbol_map_entry_t* entry = bindings->pending.entries;
  while (entry->name_id == LOOM_STRING_ID_INVALID ||
         entry->name_id == LOOM_SYMBOL_MAP_TOMBSTONE) {
    ++entry;
  }
  return loom_bytecode_reader_emit_invalid_field(
      decoder, IREE_SV("TYPES"), IREE_SV("reference"), 0,
      IREE_SV("declaration"),
      bindings->values[entry->symbol_id].bytecode_offset,
      IREE_SV("managed_type_declaration_name_is_not_defined"));
}
