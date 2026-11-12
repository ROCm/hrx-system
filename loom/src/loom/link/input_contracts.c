// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/input_contracts.h"

#include "loom/ir/intern_table.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/op_defs.h"

// All rows are canonical source-coordinate records, independent of SSA IDs.
typedef struct loom_link_input_contract_probe_t {
  // Previously retained unique contracts in interner ordinal order.
  const loom_input_contract_t* const* contracts;
  // Complete candidate row allocated after a restorable checkpoint.
  const loom_input_contract_t* candidate;
} loom_link_input_contract_probe_t;

static bool loom_link_input_contract_equal(const void* opaque, uint32_t index) {
  const loom_link_input_contract_probe_t* probe = opaque;
  const loom_input_contract_t* existing = probe->contracts[index];
  return existing->count == probe->candidate->count &&
         memcmp(existing->values, probe->candidate->values,
                existing->count * sizeof(*existing->values)) == 0;
}

static iree_status_t loom_link_project_input_contracts_impl(
    const loom_module_t* module, loom_link_module_index_symbol_t* symbols,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch,
    const loom_input_contract_t*** out_contracts, uint32_t* out_count) {
  const loom_input_contract_t** contracts = NULL;
  loom_intern_table_t index;
  IREE_RETURN_IF_ERROR(loom_intern_table_initialize(scratch, 0, &index));
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (symbol->kind != LOOM_SYMBOL_TEMPLATE_DECL &&
        symbol->kind != LOOM_SYMBOL_TEMPLATE_DEF &&
        symbol->kind != LOOM_SYMBOL_TEMPLATE_UKERNEL) {
      continue;
    }
    const loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    uint16_t argument_count = 0;
    const loom_value_id_t* arguments =
        loom_func_like_arg_ids(function, &argument_count);
    const loom_value_id_t* inherited = NULL;
    const loom_symbol_ref_t family = loom_func_like_template_family(function);
    if (loom_symbol_ref_is_valid(family)) {
      const loom_func_like_t declaration = loom_func_like_cast(
          module, module->symbols.entries[family.symbol_id].defining_op);
      uint16_t inherited_count = 0;
      inherited = loom_func_like_arg_ids(declaration, &inherited_count);
      if (inherited_count != argument_count) {
        inherited = NULL;
      }
    }
    const uint32_t count = loom_input_contract_project(
        module, arguments, argument_count, inherited, NULL);
    if (!count) {
      continue;
    }
    if (!contracts) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(scratch, module->symbols.count,
                                    sizeof(*contracts), (void**)&contracts));
    }
    const iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(arena);
    loom_input_contract_t* candidate = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        arena, sizeof(*candidate) + count * sizeof(*candidate->values),
        (void**)&candidate));
    *candidate = (loom_input_contract_t){.count = count};
    loom_input_contract_project(module, arguments, argument_count, inherited,
                                candidate->values);
    uint32_t hash = loom_structural_hash_initialize();
    for (uint32_t j = 0; j < count; ++j) {
      const loom_input_requirement_t row = candidate->values[j];
      hash = loom_structural_hash_mix_u32(hash, row.argument);
      hash = loom_structural_hash_mix_u32(hash, row.axis);
      hash = loom_structural_hash_mix_u32(hash, row.expected_is_argument);
      hash = loom_structural_hash_mix_u64(hash, row.expected);
    }
    hash = loom_structural_hash_finalize(hash);
    const loom_link_input_contract_probe_t query = {
        .contracts = contracts,
        .candidate = candidate,
    };
    loom_intern_probe_t probe = loom_intern_table_probe(
        &index, hash, loom_link_input_contract_equal, &query);
    if (probe.index != UINT32_MAX) {
      symbols[i].input_contract_ordinal_plus_one = probe.index + 1;
      iree_arena_checkpoint_restore(&checkpoint);
    } else {
      IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
          scratch, &index, hash, 1, &probe.slot));
      const uint32_t ordinal = (uint32_t)index.count;
      contracts[ordinal] = candidate;
      loom_intern_table_insert(&index, probe.slot, hash, ordinal);
      symbols[i].input_contract_ordinal_plus_one = ordinal + 1;
    }
  }
  *out_count = (uint32_t)index.count;
  if (index.count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, index.count, sizeof(**out_contracts), (void**)out_contracts));
    memcpy(*out_contracts, contracts, index.count * sizeof(*contracts));
  }
  return iree_ok_status();
}

iree_status_t loom_link_project_input_contracts(
    const loom_module_t* module, loom_link_module_index_symbol_t* symbols,
    iree_arena_allocator_t* arena, const loom_input_contract_t*** out_contracts,
    uint32_t* out_count) {
  *out_contracts = NULL;
  *out_count = 0;
  iree_arena_allocator_t scratch;
  iree_arena_initialize(arena->block_pool, &scratch);
  const iree_status_t status = loom_link_project_input_contracts_impl(
      module, symbols, arena, &scratch, out_contracts, out_count);
  iree_arena_deinitialize(&scratch);
  return status;
}
