// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_object.h"

#include <inttypes.h>
#include <string.h>

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"

static bool loom_aie2p_program_object_is_code_section(
    const loom_native_section_contribution_t* section) {
  return section->storage == LOOM_NATIVE_SECTION_STORAGE_CONTENTS &&
         section->access == (LOOM_NATIVE_SECTION_ACCESS_READ |
                             LOOM_NATIVE_SECTION_ACCESS_EXECUTE);
}

static iree_status_t loom_aie2p_program_object_validate_helper(
    const loom_aie2p_leaf_contribution_t* helper) {
  const loom_aie2p_leaf_realization_t* realization = &helper->realization;
  const loom_aie2p_leaf_capability_flags_t unsupported =
      LOOM_AIE2P_LEAF_CAPABILITY_FLAG_RESOURCE_IMPORTS |
      LOOM_AIE2P_LEAF_CAPABILITY_FLAG_READ_ONLY_DATA |
      LOOM_AIE2P_LEAF_CAPABILITY_FLAG_FUNCTION_STORAGE |
      LOOM_AIE2P_LEAF_CAPABILITY_FLAG_MATERIALIZED_SPILLS;
  if (iree_any_bit_set(realization->capability_flags, unsupported) ||
      realization->resource_import_count != 0 ||
      realization->read_only_data_count != 0 ||
      realization->storage_domain_count != 0 ||
      realization->spill.byte_length != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P direct helper must be a spill-free code-only leaf");
  }
  if (helper->object.section_count != 1 ||
      !loom_aie2p_program_object_is_code_section(&helper->object.sections[0])) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P direct helper must contribute exactly one code section");
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_object_resolve_calls(
    loom_native_object_symbol_t* symbols, iree_host_size_t symbol_count,
    loom_native_object_fixup_t* fixups, iree_host_size_t fixup_count) {
  for (iree_host_size_t i = 0; i < fixup_count; ++i) {
    loom_native_object_fixup_t* fixup = &fixups[i];
    if (fixup->relocation_kind !=
        LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_CALL_ABSOLUTE) {
      continue;
    }
    loom_native_object_symbol_t* target = &symbols[fixup->target_symbol_index];
    if (target->section_contribution_index != IREE_HOST_SIZE_MAX) {
      continue;
    }
    iree_host_size_t definition_index = IREE_HOST_SIZE_MAX;
    for (iree_host_size_t j = 0; j < symbol_count; ++j) {
      const loom_native_object_symbol_t* candidate = &symbols[j];
      if (candidate->section_contribution_index == IREE_HOST_SIZE_MAX ||
          candidate->kind != target->kind ||
          !iree_string_view_equal(candidate->name, target->name)) {
        continue;
      }
      if (definition_index != IREE_HOST_SIZE_MAX) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AIE2P direct call target `%.*s` has multiple definitions",
            (int)target->name.size, target->name.data);
      }
      definition_index = j;
    }
    if (definition_index == IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "AIE2P direct call target `%.*s` is undefined",
                              (int)target->name.size, target->name.data);
    }
    fixup->target_symbol_index = definition_index;
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_object_merge(
    const loom_aie2p_leaf_contribution_t* contributions,
    iree_host_size_t contribution_count, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_contribution_t* out_program) {
  IREE_ASSERT_ARGUMENT(contributions || contribution_count == 0);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = (loom_aie2p_leaf_contribution_t){0};
  if (contribution_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P tile program requires an entry function");
  }

  const loom_aie2p_leaf_contribution_t* entry = &contributions[0];
  if (entry->realization.entry_symbol_index >= entry->object.symbol_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P entry symbol is invalid");
  }
  const loom_native_object_symbol_t* entry_symbol =
      &entry->object.symbols[entry->realization.entry_symbol_index];
  if (entry_symbol->section_contribution_index >= entry->object.section_count ||
      !loom_aie2p_program_object_is_code_section(
          &entry->object.sections[entry_symbol->section_contribution_index])) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P entry symbol must name executable code");
  }
  const iree_string_view_t code_section_name =
      entry->object.sections[entry_symbol->section_contribution_index]
          .section_name;

  iree_host_size_t section_count = 0;
  iree_host_size_t symbol_count = 0;
  iree_host_size_t fixup_count = 0;
  for (iree_host_size_t i = 0; i < contribution_count; ++i) {
    const loom_aie2p_leaf_contribution_t* contribution = &contributions[i];
    if (!iree_string_view_equal(contribution->realization.target_identity,
                                entry->realization.target_identity) ||
        !iree_string_view_equal(contribution->realization.abi_identity,
                                entry->realization.abi_identity)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AIE2P tile program contributions disagree on target or ABI");
    }
    if (i != 0) {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_program_object_validate_helper(contribution));
    }
    if (!iree_host_size_checked_add(section_count,
                                    contribution->object.section_count,
                                    &section_count) ||
        !iree_host_size_checked_add(
            symbol_count, contribution->object.symbol_count, &symbol_count) ||
        !iree_host_size_checked_add(
            fixup_count, contribution->object.fixup_count, &fixup_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P tile program object size overflows");
    }
  }

  loom_native_section_contribution_t* sections = NULL;
  loom_native_object_symbol_t* symbols = NULL;
  loom_native_object_fixup_t* fixups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, section_count, sizeof(*sections), (void**)&sections));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, symbol_count, sizeof(*symbols), (void**)&symbols));
  if (fixup_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, fixup_count, sizeof(*fixups), (void**)&fixups));
  }

  iree_host_size_t section_start = 0;
  iree_host_size_t symbol_start = 0;
  iree_host_size_t fixup_start = 0;
  uint64_t code_byte_length = 0;
  uint64_t code_alignment = 1;
  loom_aie2p_register_unit_set_t register_writes = {0};
  loom_aie2p_leaf_capability_flags_t capability_flags = 0;
  for (iree_host_size_t i = 0; i < contribution_count; ++i) {
    const loom_aie2p_leaf_contribution_t* contribution = &contributions[i];
    for (iree_host_size_t j = 0; j < contribution->object.section_count; ++j) {
      sections[section_start + j] = contribution->object.sections[j];
      if (!loom_aie2p_program_object_is_code_section(
              &sections[section_start + j])) {
        continue;
      }
      sections[section_start + j].section_name = code_section_name;
      const uint64_t alignment = iree_max(
          (uint64_t)1, sections[section_start + j].contribution_alignment);
      if (!iree_checked_align_u64(code_byte_length, alignment,
                                  &code_byte_length) ||
          !iree_checked_add_u64(
              code_byte_length,
              sections[section_start + j].contents.data_length,
              &code_byte_length)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AIE2P tile program code size overflows");
      }
      code_alignment = iree_max(code_alignment, alignment);
    }
    for (iree_host_size_t j = 0; j < contribution->object.symbol_count; ++j) {
      symbols[symbol_start + j] = contribution->object.symbols[j];
      if (symbols[symbol_start + j].section_contribution_index !=
          IREE_HOST_SIZE_MAX) {
        symbols[symbol_start + j].section_contribution_index += section_start;
      }
    }
    for (iree_host_size_t j = 0; j < contribution->object.fixup_count; ++j) {
      fixups[fixup_start + j] = contribution->object.fixups[j];
      fixups[fixup_start + j].section_contribution_index += section_start;
      fixups[fixup_start + j].target_symbol_index += symbol_start;
    }
    for (iree_host_size_t word = 0;
         word < IREE_ARRAYSIZE(register_writes.words); ++word) {
      register_writes.words[word] |=
          contribution->realization.register_writes.words[word];
    }
    capability_flags |= contribution->realization.capability_flags;
    section_start += contribution->object.section_count;
    symbol_start += contribution->object.symbol_count;
    fixup_start += contribution->object.fixup_count;
  }
  IREE_ASSERT_EQ(section_start, section_count);
  IREE_ASSERT_EQ(symbol_start, symbol_count);
  IREE_ASSERT_EQ(fixup_start, fixup_count);
  if (code_byte_length > LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P tile program requires %" PRIu64 " code bytes; capacity is %u",
        code_byte_length, LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE);
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_program_object_resolve_calls(
      symbols, symbol_count, fixups, fixup_count));

  *out_program = *entry;
  out_program->object = (loom_native_object_contribution_t){
      .sections = sections,
      .section_count = section_count,
      .symbols = symbols,
      .symbol_count = symbol_count,
      .fixups = fixups,
      .fixup_count = fixup_count,
  };
  out_program->realization.code = (loom_aie2p_leaf_storage_requirement_t){
      .byte_length = code_byte_length,
      .minimum_alignment = code_alignment,
  };
  out_program->realization.register_writes = register_writes;
  out_program->realization.capability_flags = capability_flags;
  return iree_ok_status();
}
