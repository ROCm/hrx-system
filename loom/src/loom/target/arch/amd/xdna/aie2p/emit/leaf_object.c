// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_object.h"

#include <string.h>

#include "loom/codegen/low/diagnostics.h"

static iree_status_t loom_aie2p_leaf_object_copy_section_name(
    iree_string_view_t function_name, iree_arena_allocator_t* arena,
    iree_string_view_t* out_section_name, iree_string_view_t* out_symbol_name) {
  const iree_string_view_t prefix = IREE_SV(".text.");
  iree_host_size_t section_name_length = 0;
  if (!iree_host_size_checked_add(prefix.size, function_name.size,
                                  &section_name_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P leaf section name is too long");
  }
  char* section_name_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, section_name_length,
                                           (void**)&section_name_data));
  memcpy(section_name_data, prefix.data, prefix.size);
  memcpy(section_name_data + prefix.size, function_name.data,
         function_name.size);
  *out_section_name =
      iree_make_string_view(section_name_data, section_name_length);
  *out_symbol_name = iree_make_string_view(section_name_data + prefix.size,
                                           function_name.size);
  return iree_ok_status();
}

iree_status_t loom_aie2p_leaf_object_emit(
    const loom_aie2p_bundle_plan_t* plan, iree_arena_allocator_t* arena,
    loom_native_object_contribution_t* out_object) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(plan->frame);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_object);
  *out_object = (loom_native_object_contribution_t){0};

  uint8_t* code = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, plan->encoded_byte_length, (void**)&code));
  iree_host_size_t code_offset = 0;
  for (iree_host_size_t i = 0; i < plan->bundle_count; ++i) {
    const loom_aie2p_planned_bundle_t* bundle = &plan->bundles[i];
    loom_aie2p_encoded_slot_t
        encoded_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
    for (uint8_t j = 0; j < bundle->slot_count; ++j) {
      encoded_slots[j] = plan->slots[bundle->slot_start + j].encoded_slot;
    }
    loom_aie2p_encoding_packet_t packet;
    IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_bundle(
        bundle->format, encoded_slots, bundle->slot_count, &packet));
    IREE_ASSERT(code_offset + packet.data_length <= plan->encoded_byte_length);
    memcpy(code + code_offset, packet.data, packet.data_length);
    code_offset += packet.data_length;
  }
  IREE_ASSERT(code_offset == plan->encoded_byte_length);

  iree_string_view_t section_name;
  iree_string_view_t symbol_name;
  IREE_RETURN_IF_ERROR(loom_aie2p_leaf_object_copy_section_name(
      loom_low_diagnostic_function_name(plan->frame->module,
                                        plan->frame->function_op),
      arena, &section_name, &symbol_name));

  loom_native_section_contribution_t* section = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*section), (void**)&section));
  *section = (loom_native_section_contribution_t){
      .section_name = section_name,
      .section_type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      .section_flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                       LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
      .contribution_alignment = 16,
      .contents = iree_make_const_byte_span(code, plan->encoded_byte_length),
  };

  loom_native_object_symbol_t* symbol = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*symbol), (void**)&symbol));
  *symbol = (loom_native_object_symbol_t){
      .name = symbol_name,
      .section_contribution_index = 0,
      .section_offset = 0,
      .size = plan->encoded_byte_length,
      .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
      .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };

  *out_object = (loom_native_object_contribution_t){
      .sections = section,
      .section_count = 1,
      .symbols = symbol,
      .symbol_count = 1,
  };
  return iree_ok_status();
}
