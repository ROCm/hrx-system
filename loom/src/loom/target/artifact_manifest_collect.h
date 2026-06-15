// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Collects target-neutral artifact manifest facts from Loom IR analyses.

#ifndef LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_H_
#define LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_dependencies.h"
#include "loom/ir/ir.h"
#include "loom/target/artifact_manifest.h"
#include "loom/target/entry_selection.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_target_artifact_manifest_collect_flags_t;

enum loom_target_artifact_manifest_collect_flag_bits_e {
  // Indicates that artifact_byte_length carries the emitted artifact byte
  // length. The collector preserves zero when this flag is set.
  LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_FLAG_ARTIFACT_BYTE_LENGTH = 1u << 0,
};

typedef struct loom_target_artifact_manifest_collect_options_t {
  // Selected manifest detail mode.
  loom_target_artifact_manifest_mode_t mode;

  // Bitfield selecting optional collector inputs that are present.
  loom_target_artifact_manifest_collect_flags_t flags;

  // Emitted artifact byte length when ARTIFACT_BYTE_LENGTH is set.
  uint64_t artifact_byte_length;

  // Optional artifact name supplied by the emitter or packager.
  iree_string_view_t artifact_name;

  // Emitted artifact format. UNKNOWN falls back to the first selected entry's
  // target snapshot format when entries are present.
  loom_target_artifact_format_t artifact_format;
} loom_target_artifact_manifest_collect_options_t;

// Initializes collection options with manifest collection disabled.
void loom_target_artifact_manifest_collect_options_initialize(
    loom_target_artifact_manifest_collect_options_t* out_options);

// Collects a manifest for selected emitted entries into arena-owned storage.
//
// NONE mode only clears |out_manifest|. SUMMARY mode collects cheap structural
// facts already present in selected entries, target facts, and the symbol
// dependency table. DETAILS and ANALYSIS modes additionally collect
// per-parameter rows from function signatures. Expensive artifact byte scans,
// hashing, disassembly, target-specific static analysis, and pass reports are
// intentionally outside this shared collector.
//
// Manifest arrays are allocated from |arena|. String views borrow storage from
// |module|, |entries|, and static literals and must not outlive them.
iree_status_t loom_target_artifact_manifest_collect_from_entries(
    const loom_module_t* module, loom_target_entry_list_t entries,
    const loom_symbol_dependency_table_t* dependency_table,
    const loom_target_artifact_manifest_collect_options_t* options,
    iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_t* out_manifest);

// Collects and formats a manifest for selected emitted entries.
//
// NONE mode returns empty JSON contents. Other modes build the symbol
// dependency table needed by global-use reporting, collect target-neutral
// manifest facts, and format JSON into allocator-owned contents.
iree_status_t loom_target_artifact_manifest_collect_json_from_entries(
    const loom_module_t* module, loom_target_entry_list_t entries,
    const loom_target_artifact_manifest_collect_options_t* options,
    iree_arena_allocator_t* arena, iree_allocator_t allocator,
    loom_target_artifact_manifest_json_t* out_json);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_H_
