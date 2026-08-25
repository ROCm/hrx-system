// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lazy template applicability projection from validated bytecode metadata.

#ifndef LOOM_LINK_BYTECODE_TEMPLATE_CONTRACT_H_
#define LOOM_LINK_BYTECODE_TEMPLATE_CONTRACT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/format/bytecode/index.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source-domain applicability contract for one bytecode template provider.
typedef struct loom_link_bytecode_template_contract_t {
  // Representation-independent provider applicability.
  loom_template_provider_contract_t provider;

  // Source SYMBOLS ordinal of the target witness, or UINT32_MAX.
  uint32_t target_symbol_ordinal;
} loom_link_bytecode_template_contract_t;

// Per-bytecode-module lazy applicability cache.
typedef struct loom_link_bytecode_template_contract_reader_t {
  // Borrowed complete bytecode file contents.
  iree_const_byte_span_t bytecode;

  // Finalized context owning operation and attribute descriptors.
  loom_context_t* context;

  // Borrowed validated source module metadata.
  const loom_bytecode_module_metadata_t* metadata;

  // Persistent storage for projected contracts and target facts.
  iree_arena_allocator_t* arena;

  // Provider contracts indexed by source SYMBOLS ordinal.
  struct {
    // Arena-owned contract pointers; entries are populated on first use.
    const loom_link_bytecode_template_contract_t** values;

    // Number of source symbol slots.
    iree_host_size_t count;
  } contracts;

  // Target fact projections indexed by source SYMBOLS ordinal.
  struct {
    // Arena-owned fact pointers; NULL may be a projected target declaration.
    const loom_target_facts_t** values;

    // Arena-owned bits distinguishing unprojected entries from NULL results.
    uint64_t* projected_words;
  } targets;
} loom_link_bytecode_template_contract_reader_t;

// Initializes an empty lazy reader over one validated bytecode module.
//
// The reader borrows every argument for its lifetime. It creates no IR and
// never reads provider implementation bodies.
iree_status_t loom_link_bytecode_template_contract_reader_initialize(
    iree_const_byte_span_t bytecode, loom_context_t* context,
    const loom_bytecode_module_metadata_t* metadata,
    iree_arena_allocator_t* arena,
    loom_link_bytecode_template_contract_reader_t* out_reader);

// Returns one cached provider applicability contract.
iree_status_t loom_link_bytecode_template_contract_reader_load(
    loom_link_bytecode_template_contract_reader_t* reader,
    uint32_t source_symbol_ordinal,
    const loom_link_bytecode_template_contract_t** out_contract);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_BYTECODE_TEMPLATE_CONTRACT_H_
