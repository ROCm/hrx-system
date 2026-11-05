// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_object.h"

#include <string.h>

#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"
#include "loom/util/adaptive_sort.h"

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

// Direct helpers are module-internal leaves whose addresses are used only by
// calls and their own branches. Distinct symbols can therefore share a physical
// body when its bytes and relocations agree. Symbol names remain distinct;
// imported names must still agree because they identify external definitions.
static int loom_aie2p_program_object_compare_helpers(
    const loom_aie2p_leaf_contribution_t* lhs,
    const loom_aie2p_leaf_contribution_t* rhs) {
  const loom_native_object_contribution_t* a = &lhs->object;
  const loom_native_object_contribution_t* b = &rhs->object;
  if (a->symbol_count != b->symbol_count) {
    return a->symbol_count < b->symbol_count ? -1 : 1;
  }
  if (a->fixup_count != b->fixup_count) {
    return a->fixup_count < b->fixup_count ? -1 : 1;
  }
  if (a->sections[0].contribution_alignment !=
      b->sections[0].contribution_alignment) {
    return a->sections[0].contribution_alignment <
                   b->sections[0].contribution_alignment
               ? -1
               : 1;
  }
  if (a->sections[0].contents.data_length !=
      b->sections[0].contents.data_length) {
    return a->sections[0].contents.data_length <
                   b->sections[0].contents.data_length
               ? -1
               : 1;
  }
  int comparison =
      memcmp(a->sections[0].contents.data, b->sections[0].contents.data,
             a->sections[0].contents.data_length);
  if (comparison != 0) {
    return comparison;
  }
  for (iree_host_size_t i = 0; i < a->symbol_count; ++i) {
    const loom_native_object_symbol_t* x = &a->symbols[i];
    const loom_native_object_symbol_t* y = &b->symbols[i];
    if (x->section_contribution_index != y->section_contribution_index) {
      return x->section_contribution_index < y->section_contribution_index ? -1
                                                                           : 1;
    }
    if (x->section_offset != y->section_offset) {
      return x->section_offset < y->section_offset ? -1 : 1;
    }
    if (x->size != y->size) {
      return x->size < y->size ? -1 : 1;
    }
    if (x->binding != y->binding) {
      return x->binding < y->binding ? -1 : 1;
    }
    if (x->visibility != y->visibility) {
      return x->visibility < y->visibility ? -1 : 1;
    }
    if (x->kind != y->kind) {
      return x->kind < y->kind ? -1 : 1;
    }
    if (x->section_contribution_index == IREE_HOST_SIZE_MAX) {
      comparison = iree_string_view_compare(x->name, y->name);
      if (comparison != 0) {
        return comparison;
      }
    }
  }
  for (iree_host_size_t i = 0; i < a->fixup_count; ++i) {
    const loom_native_object_fixup_t* x = &a->fixups[i];
    const loom_native_object_fixup_t* y = &b->fixups[i];
    if (x->section_contribution_index != y->section_contribution_index) {
      return x->section_contribution_index < y->section_contribution_index ? -1
                                                                           : 1;
    }
    if (x->section_offset != y->section_offset) {
      return x->section_offset < y->section_offset ? -1 : 1;
    }
    if (x->relocation_kind != y->relocation_kind) {
      return x->relocation_kind < y->relocation_kind ? -1 : 1;
    }
    if (x->target_symbol_index != y->target_symbol_index) {
      return x->target_symbol_index < y->target_symbol_index ? -1 : 1;
    }
    if (x->addend != y->addend) {
      return x->addend < y->addend ? -1 : 1;
    }
  }
  return 0;
}

static bool loom_aie2p_program_object_helper_less(
    const loom_aie2p_leaf_contribution_t* contributions,
    const iree_host_size_t* lhs, const iree_host_size_t* rhs) {
  const int comparison = loom_aie2p_program_object_compare_helpers(
      &contributions[*lhs], &contributions[*rhs]);
  // Equivalent bodies retain their first input as the canonical contribution,
  // independently of the sort's ordering of otherwise equal elements.
  return comparison != 0 ? comparison < 0 : *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_aie2p_program_object_sort_helpers,
                                       iree_host_size_t,
                                       const loom_aie2p_leaf_contribution_t*,
                                       loom_aie2p_program_object_helper_less)

typedef struct loom_aie2p_program_object_placement_t {
  // First section in the merged object, assigned only for canonical bodies.
  iree_host_size_t section_start;
  // First symbol in the merged object; every contribution keeps its names.
  iree_host_size_t symbol_start;
} loom_aie2p_program_object_placement_t;

// Every emitted leaf entry has the same 16-byte alignment. Alignment before
// the first helper is fixed by the entry, and every helper except the final one
// pays the padding implied by its own byte length. Leaving the helper with the
// largest such padding last therefore minimizes the complete program length
// while preserving the relative order of every other helper.
static iree_host_size_t loom_aie2p_program_object_select_tail(
    const loom_aie2p_leaf_contribution_t* contributions,
    const iree_host_size_t* canonical_indices,
    iree_host_size_t contribution_count, uint64_t code_alignment) {
  iree_host_size_t tail_index = 0;
  uint64_t maximum_padding = 0;
  for (iree_host_size_t i = 1; i < contribution_count; ++i) {
    if (canonical_indices[i] != i) {
      continue;
    }
    const loom_native_section_contribution_t* code =
        &contributions[i].object.sections[0];
    const uint64_t alignment =
        iree_max((uint64_t)1, code->contribution_alignment);
    IREE_ASSERT_EQ(alignment, code_alignment,
                   "AIE2P leaf code contribution alignment");
    const uint64_t remainder = code->contents.data_length % alignment;
    const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
    if (padding >= maximum_padding) {
      maximum_padding = padding;
      tail_index = i;
    }
  }
  return tail_index;
}

static iree_host_size_t loom_aie2p_program_object_layout_index(
    iree_host_size_t layout_index, iree_host_size_t contribution_count,
    iree_host_size_t tail_index) {
  if (layout_index == 0 || tail_index == contribution_count - 1 ||
      layout_index < tail_index) {
    return layout_index;
  }
  return layout_index == contribution_count - 1 ? tail_index : layout_index + 1;
}

iree_status_t loom_aie2p_program_object_merge(
    const loom_aie2p_leaf_contribution_t* contributions,
    iree_host_size_t contribution_count, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_aie2p_program_object_t* out_program) {
  IREE_ASSERT_ARGUMENT(contributions || contribution_count == 0);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = (loom_aie2p_program_object_t){0};
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
  const uint64_t entry_code_alignment =
      iree_max((uint64_t)1,
               entry->object.sections[entry_symbol->section_contribution_index]
                   .contribution_alignment);

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
  }

  iree_host_size_t* helper_order = NULL;
  iree_host_size_t* canonical_indices = NULL;
  loom_aie2p_program_object_placement_t* placements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, contribution_count,
                                sizeof(*helper_order), (void**)&helper_order));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, contribution_count, sizeof(*canonical_indices),
      (void**)&canonical_indices));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, contribution_count,
                                sizeof(*placements), (void**)&placements));
  for (iree_host_size_t i = 0; i < contribution_count; ++i) {
    helper_order[i] = i;
    canonical_indices[i] = i;
  }
  // Sorting places equivalent helpers together without comparing every pair.
  // Only the index projection moves; physical layout retains input order.
  loom_aie2p_program_object_sort_helpers(contributions, helper_order + 1,
                                         contribution_count - 1);
  for (iree_host_size_t i = 2; i < contribution_count; ++i) {
    const iree_host_size_t current = helper_order[i];
    const iree_host_size_t previous = helper_order[i - 1];
    if (loom_aie2p_program_object_compare_helpers(
            &contributions[current], &contributions[previous]) == 0) {
      canonical_indices[current] = canonical_indices[previous];
    }
  }

  iree_host_size_t section_count = 0;
  iree_host_size_t symbol_count = 0;
  iree_host_size_t fixup_count = 0;
  for (iree_host_size_t i = 0; i < contribution_count; ++i) {
    const loom_aie2p_leaf_contribution_t* contribution = &contributions[i];
    placements[i] = (loom_aie2p_program_object_placement_t){
        .symbol_start = symbol_count,
    };
    const bool shares_code = canonical_indices[i] != i;
    if (!iree_host_size_checked_add(
            section_count, shares_code ? 0 : contribution->object.section_count,
            &section_count) ||
        !iree_host_size_checked_add(
            symbol_count, contribution->object.symbol_count, &symbol_count) ||
        !iree_host_size_checked_add(
            fixup_count, shares_code ? 0 : contribution->object.fixup_count,
            &fixup_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P tile program object size overflows");
    }
  }

  const iree_host_size_t tail_contribution_index =
      loom_aie2p_program_object_select_tail(contributions, canonical_indices,
                                            contribution_count,
                                            entry_code_alignment);

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
  iree_host_size_t fixup_start = 0;
  uint64_t code_byte_length = 0;
  uint64_t code_alignment = 1;
  loom_aie2p_register_unit_set_t register_writes = {0};
  loom_aie2p_leaf_capability_flags_t capability_flags = 0;
  for (iree_host_size_t layout_index = 0; layout_index < contribution_count;
       ++layout_index) {
    const iree_host_size_t contribution_index =
        loom_aie2p_program_object_layout_index(layout_index, contribution_count,
                                               tail_contribution_index);
    const loom_aie2p_leaf_contribution_t* contribution =
        &contributions[contribution_index];
    if (canonical_indices[contribution_index] != contribution_index) {
      continue;
    }
    placements[contribution_index].section_start = section_start;
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
    for (iree_host_size_t j = 0; j < contribution->object.fixup_count; ++j) {
      fixups[fixup_start + j] = contribution->object.fixups[j];
      fixups[fixup_start + j].section_contribution_index += section_start;
      fixups[fixup_start + j].target_symbol_index +=
          placements[contribution_index].symbol_start;
    }
    section_start += contribution->object.section_count;
    fixup_start += contribution->object.fixup_count;
  }
  for (iree_host_size_t i = 0; i < contribution_count; ++i) {
    const loom_aie2p_leaf_contribution_t* contribution = &contributions[i];
    const loom_aie2p_program_object_placement_t* placement = &placements[i];
    const iree_host_size_t canonical_section_start =
        placements[canonical_indices[i]].section_start;
    for (iree_host_size_t j = 0; j < contribution->object.symbol_count; ++j) {
      loom_native_object_symbol_t* symbol =
          &symbols[placement->symbol_start + j];
      *symbol = contribution->object.symbols[j];
      if (symbol->section_contribution_index != IREE_HOST_SIZE_MAX) {
        symbol->section_contribution_index += canonical_section_start;
      }
    }
    for (iree_host_size_t word = 0;
         word < IREE_ARRAYSIZE(register_writes.words); ++word) {
      register_writes.words[word] |=
          contribution->realization.register_writes.words[word];
    }
    capability_flags |= contribution->realization.capability_flags;
  }
  IREE_ASSERT_EQ(section_start, section_count);
  IREE_ASSERT_EQ(fixup_start, fixup_count);
  IREE_RETURN_IF_ERROR(loom_aie2p_program_object_resolve_calls(
      symbols, symbol_count, fixups, fixup_count));

  out_program->canonical_indices = canonical_indices;
  out_program->contribution = *entry;
  out_program->contribution.object = (loom_native_object_contribution_t){
      .sections = sections,
      .section_count = section_count,
      .symbols = symbols,
      .symbol_count = symbol_count,
      .fixups = fixups,
      .fixup_count = fixup_count,
  };
  out_program->contribution.realization.code =
      (loom_aie2p_leaf_storage_requirement_t){
          .byte_length = code_byte_length,
          .minimum_alignment = code_alignment,
      };
  out_program->contribution.realization.register_writes = register_writes;
  out_program->contribution.realization.capability_flags = capability_flags;
  return iree_ok_status();
}
