// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Incremental canonical module.import construction for the Loom linker.

#ifndef LOOM_LINK_PROVIDER_IMPORT_SINK_H_
#define LOOM_LINK_PROVIDER_IMPORT_SINK_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// One linker-owned projected provider-import row.
typedef struct loom_link_provider_import_sink_row_t {
  // Target-module provider string ID.
  loom_string_id_t provider_id;
  // Slice in the sink's flat target-anchor array.
  struct {
    // First target anchor ordinal.
    uint32_t first;
    // Number of target anchors.
    uint32_t count;
  } anchors;
  // Copied source comments retained until output construction.
  struct {
    // Linker-arena-owned normalized comment payloads.
    const iree_string_view_t* values;
    // Number of comment payloads.
    uint16_t count;
  } comments;
  // True when the source import had authored leading vertical separation.
  bool leading_blank_line;
} loom_link_provider_import_sink_row_t;

// Allocation-bounded provider-import accumulator owned by one linker.
typedef struct loom_link_provider_import_sink_t {
  // Linked output module receiving canonical import operations.
  loom_module_t* target_module;
  // Linker-lifetime arena owning projected rows, anchors, and comments.
  iree_arena_allocator_t* arena;
  // Projected provider-import rows.
  struct {
    // Arena-owned row array.
    loom_link_provider_import_sink_row_t* values;
    // Number of appended rows.
    iree_host_size_t count;
    // Exact maximum number of projected rows.
    iree_host_size_t capacity;
  } rows;
  // Target symbol references from projected rows.
  struct {
    // Arena-owned flat target-anchor array.
    loom_symbol_ref_t* values;
    // Number of appended target anchors.
    iree_host_size_t count;
    // Exact maximum number of projected target anchors.
    iree_host_size_t capacity;
  } anchors;
} loom_link_provider_import_sink_t;

// Initializes |out_sink| with exact projected row and anchor capacities.
//
// Existing module.import operations already present in |target_module| are
// borrowed directly during finish and do not consume projected capacity.
iree_status_t loom_link_provider_import_sink_initialize(
    loom_module_t* target_module, iree_arena_allocator_t* arena,
    iree_host_size_t row_capacity, iree_host_size_t anchor_capacity,
    loom_link_provider_import_sink_t* out_sink);

// Appends one projected import row after its anchors have been mapped into the
// target symbol domain. Provider and comment payloads are copied before return.
iree_status_t loom_link_provider_import_sink_append(
    loom_link_provider_import_sink_t* sink, iree_string_view_t provider,
    loom_symbol_ref_array_t target_anchors, iree_string_view_list_t comments,
    bool leading_blank_line);

// Replaces existing and projected imports with one canonical operation per
// provider key. Anchor sets are unioned and sorted by final target symbol name;
// comments are retained in deterministic lexical row order.
iree_status_t loom_link_provider_import_sink_finish(
    loom_link_provider_import_sink_t* sink);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_PROVIDER_IMPORT_SINK_H_
