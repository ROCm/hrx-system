// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_SYMBOL_REFERENCE_SUMMARY_H_
#define LOOM_ANALYSIS_SYMBOL_REFERENCE_SUMMARY_H_

#include "loom/analysis/symbol_references.h"

#ifdef __cplusplus
extern "C" {
#endif

static_assert(LOOM_TYPE_COUNT_ == 17,
              "update symbol-bearing type classification for new kinds");

// Shallow structural classification, without examining nested payloads. An
// untyped register carries only target-owned words, not a semantic value type.
static inline bool loom_symbol_reference_type_may_contain_ref(
    loom_type_t type) {
  if (!loom_type_kind_is_valid(loom_type_kind(type))) {
    return false;
  }
  if (loom_type_has_static_encoding(type)) {
    return true;
  }
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_FUNCTION:
    case LOOM_TYPE_DIALECT:
    case LOOM_TYPE_PARAMETERIZED:
    case LOOM_TYPE_REFERENCE:
      return true;
    case LOOM_TYPE_REGISTER:
      return loom_type_register_has_value_type(type);
    default:
      return false;
  }
}

// Invocation-local structural summaries for one immutable module. Source
// symbols, regions, and operations belong to occurrence emission, not these
// summaries. All storage is borrowed from a scratch arena; no result escapes
// the table build. Queries replace the current cursor but retain built facts.
// Storage stays pinned after initialization; a failed query ends the build.
typedef struct loom_symbol_reference_summary_t {
  // Immutable owner of every borrowed structural payload and descriptor.
  const loom_module_t* module;
  // Scratch storage, distinct from the occurrence table's output arena.
  iree_arena_allocator_t* arena;
  // Hash-indexed exact-storage roots with bounded radix collision trees.
  struct {
    // Tagged roots in the active bucket array.
    uintptr_t* values;
    // Power-of-two number of bucket roots.
    iree_host_size_t capacity;
    // Number of reached exact-storage identities.
    iree_host_size_t count;
    // Retained entries owning linear-time resize enumeration.
    struct loom_symbol_reference_summary_entry_t* entries;
    // Inline buckets avoid a separate allocation for small graphs.
    uintptr_t initial[4];
  } index;
  // Active postorder frame, owned by its stable exact-storage index entry.
  struct loom_symbol_reference_summary_frame_t* construction;
  // Resolved child summaries for active construction frames.
  struct {
    // Arena-owned summary references.
    struct loom_symbol_reference_summary_ref_t* values;
    // Active reference count.
    iree_host_size_t count;
    // Allocated reference capacity.
    iree_host_size_t capacity;
  } children;
  // Output-sensitive expansion of the current query.
  struct {
    // Arena-owned traversal frames.
    struct loom_symbol_reference_summary_cursor_t* values;
    // Active cursor depth.
    iree_host_size_t count;
    // Allocated cursor capacity.
    iree_host_size_t capacity;
  } cursor;
} loom_symbol_reference_summary_t;

// Consecutive module-local references sharing occurrence metadata. References
// are borrowed until the summary arena is reset; multiplicity is preserved.
typedef struct loom_symbol_reference_summary_span_t {
  // Ordered, valid module-local target references.
  const loom_symbol_ref_t* targets;
  // Number of targets, always nonzero for a returned span.
  iree_host_size_t count;
  // Structural interface constraint for each target.
  loom_symbol_interface_flags_t interfaces;
  // Occurrence kind after composed structural transitions.
  loom_symbol_reference_occurrence_kind_t kind;
  // Dependency or availability role for every target.
  loom_symbol_reference_role_t role;
} loom_symbol_reference_summary_span_t;

void loom_symbol_reference_summary_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_symbol_reference_summary_t* summary);

// Starts a query, retaining newly discovered structural facts. Allocation and
// malformed input can fail here; advancing a completed query is infallible.
iree_status_t loom_symbol_reference_summary_query_type(
    loom_symbol_reference_summary_t* summary, loom_type_t type,
    loom_symbol_reference_occurrence_kind_t kind);
iree_status_t loom_symbol_reference_summary_query_attr(
    loom_symbol_reference_summary_t* summary, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_symbol_reference_occurrence_kind_t kind);
iree_status_t loom_symbol_reference_summary_query_encoding(
    loom_symbol_reference_summary_t* summary, uint16_t encoding_id,
    loom_symbol_reference_occurrence_kind_t kind);

// Returns the next nonempty span in structural occurrence order.
bool loom_symbol_reference_summary_next(
    loom_symbol_reference_summary_t* summary,
    loom_symbol_reference_summary_span_t* out_span);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_SYMBOL_REFERENCE_SUMMARY_H_
