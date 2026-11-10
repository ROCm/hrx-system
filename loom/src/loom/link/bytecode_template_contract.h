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

typedef struct loom_link_bytecode_template_target_cache_t
    loom_link_bytecode_template_target_cache_t;

// Decoded fields not already retained in admitted symbol metadata. Each record
// belongs to one source provider; zero initialization denotes an unread record.
typedef struct loom_link_bytecode_template_provider_projection_t {
  // Source SYMBOLS ordinal of the target witness, or UINT32_MAX.
  uint32_t target_symbol_ordinal;
  // Number of decoded conditions.
  uint16_t condition_count;
  // True only after all referenced data has been successfully projected.
  bool initialized;
  // Arena-owned resolved target conditions.
  const loom_target_condition_t* conditions;
  // Borrowed cached target projection, including NULL for declarations.
  const loom_target_facts_t* target_facts;
} loom_link_bytecode_template_provider_projection_t;

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

  // Arena-owned projections for reached target witnesses only.
  // NULL before a contract requires a target identity.
  loom_link_bytecode_template_target_cache_t* targets;
} loom_link_bytecode_template_contract_reader_t;

// Initializes an empty lazy reader over one validated bytecode module.
//
// The reader borrows every argument for its lifetime. It creates no IR and
// never reads provider implementation bodies.
void loom_link_bytecode_template_contract_reader_initialize(
    iree_const_byte_span_t bytecode, loom_context_t* context,
    const loom_bytecode_module_metadata_t* metadata,
    iree_arena_allocator_t* arena,
    loom_link_bytecode_template_contract_reader_t* out_reader);

// Returns a provider contract borrowing retained metadata and cached
// projections. The value itself needs no persistent allocation. Its referenced
// data remains valid for the reader lifetime, including across subsequent
// projections. The caller provides one zero-initialized projection per reached
// provider and reuses it only with the same source symbol and reader.
iree_status_t loom_link_bytecode_template_contract_reader_load(
    loom_link_bytecode_template_contract_reader_t* reader,
    uint32_t source_symbol_ordinal,
    loom_link_bytecode_template_provider_projection_t* projection,
    loom_link_bytecode_template_contract_t* out_contract);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_BYTECODE_TEMPLATE_CONTRACT_H_
