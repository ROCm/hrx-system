// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final sanitizer site collection.
//
// Sanitizer assertion/report operations carry executable semantics in the IR.
// Their source locations may additionally carry compact diagnostic metadata,
// but locations are optional and may be stripped. Final report site IDs are
// therefore assigned late by walking the surviving assertion/report operations
// that remain in the materialized region or function.

#ifndef LOOM_SANITIZER_SITE_COLLECTION_H_
#define LOOM_SANITIZER_SITE_COLLECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/location.h"
#include "loom/ops/op_defs.h"
#include "loom/sanitizer/site_payload.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dense final report site ID assigned by a site collection pass.
typedef uint32_t loom_sanitizer_site_id_t;
#define LOOM_SANITIZER_SITE_ID_INVALID ((loom_sanitizer_site_id_t)UINT32_MAX)

enum loom_sanitizer_site_row_flag_bits_e {
  // The row decoded a LOOM_LOCATION_TAG_SANITIZER_SITE payload.
  LOOM_SANITIZER_SITE_ROW_HAS_PAYLOAD = 1u << 0,
};
// Bitfield of loom_sanitizer_site_row_flag_bits_e values.
typedef uint32_t loom_sanitizer_site_row_flags_t;

typedef struct loom_sanitizer_site_row_t {
  // Dense ID assigned by final collection order.
  loom_sanitizer_site_id_t site_id;

  // Borrowed surviving assertion/report operation that owns this site.
  const loom_op_t* op;

  // Operation kind captured from op.
  loom_op_kind_t op_kind;

  // Original location carried by op, or LOOM_LOCATION_UNKNOWN when stripped.
  loom_location_id_t location;

  // Tagged location entry that carried payload, or LOOM_LOCATION_UNKNOWN.
  loom_location_id_t payload_location;

  // Child location under the sanitizer tag, or LOOM_LOCATION_UNKNOWN.
  loom_location_id_t source_location;

  // Row metadata flags.
  loom_sanitizer_site_row_flags_t flags;

  // Decoded sanitizer payload when LOOM_SANITIZER_SITE_ROW_HAS_PAYLOAD is set.
  loom_sanitizer_site_payload_t payload;
} loom_sanitizer_site_row_t;

typedef struct loom_sanitizer_site_collection_t {
  // Arena-owned rows in deterministic final site ID order.
  loom_sanitizer_site_row_t* rows;

  // Number of rows in rows.
  iree_host_size_t row_count;
} loom_sanitizer_site_collection_t;

// Returns true when |row| has decoded sanitizer site payload metadata.
static inline bool loom_sanitizer_site_row_has_payload(
    const loom_sanitizer_site_row_t* row) {
  return row != NULL &&
         iree_all_bits_set(row->flags, LOOM_SANITIZER_SITE_ROW_HAS_PAYLOAD);
}

// Collects final sanitizer sites from |region| in deterministic pre-order.
//
// The returned rows are allocated from |arena| and remain valid for the arena's
// lifetime. Codegen-relevant site existence comes from surviving sanitizer
// assertion/report operations. Location payloads only enrich diagnostics:
// unknown or stripped locations still produce rows, while malformed sanitizer
// payloads on surviving sites fail loudly.
iree_status_t loom_sanitizer_site_collection_build_region(
    const loom_module_t* module, loom_region_t* region,
    iree_arena_allocator_t* arena,
    loom_sanitizer_site_collection_t* out_collection);

// Collects final sanitizer sites from all root regions of |function|.
//
// Regions are visited in physical function region order. Within each region,
// operation order matches loom_walk_region(..., LOOM_WALK_PRE_ORDER, ...).
iree_status_t loom_sanitizer_site_collection_build_function(
    const loom_module_t* module, loom_func_like_t function,
    iree_arena_allocator_t* arena,
    loom_sanitizer_site_collection_t* out_collection);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_SANITIZER_SITE_COLLECTION_H_
