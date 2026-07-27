// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native object contribution records for stitched code artifacts.
//
// Backend workers describe local section bytes, symbols, and fixups relative to
// their own section contributions. Final artifact writers first assemble
// section contributions, then globalize symbol and fixup offsets through the
// section-contribution layout table. This keeps worker emission parallel and
// avoids reading arbitrary object files on the production path.

#ifndef LOOM_TARGET_EMIT_NATIVE_OBJECT_H_
#define LOOM_TARGET_EMIT_NATIVE_OBJECT_H_

#include "iree/base/api.h"
#include "loom/target/emit/native/contribution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_native_object_symbol_binding_e {
  LOOM_NATIVE_OBJECT_SYMBOL_BINDING_NONE = 0,
  LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL = 1,
  LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL = 2,
  LOOM_NATIVE_OBJECT_SYMBOL_BINDING_WEAK = 3,
} loom_native_object_symbol_binding_t;

typedef enum loom_native_object_symbol_visibility_e {
  LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT = 0,
  LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_INTERNAL = 1,
  LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_HIDDEN = 2,
  LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_PROTECTED = 3,
} loom_native_object_symbol_visibility_t;

typedef enum loom_native_object_symbol_kind_e {
  LOOM_NATIVE_OBJECT_SYMBOL_KIND_NONE = 0,
  LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION = 1,
  LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA = 2,
} loom_native_object_symbol_kind_t;

typedef struct loom_native_object_symbol_t {
  // Symbol name emitted into the final object symbol table.
  iree_string_view_t name;
  // Section contribution containing the symbol definition.
  iree_host_size_t section_contribution_index;
  // Byte offset of the symbol within the referenced section contribution.
  uint64_t section_offset;
  // Symbol byte size, or zero when the target writer cannot describe it yet.
  uint64_t size;
  // Linkage binding for the final object symbol.
  uint32_t binding;
  // Symbol visibility in the final object.
  uint32_t visibility;
  // Symbol payload kind in the final object.
  uint32_t kind;
} loom_native_object_symbol_t;

typedef struct loom_native_object_symbol_layout_t {
  // Final assembled section containing the symbol definition.
  iree_host_size_t section_index;
  // Byte offset of the symbol within the final assembled section.
  uint64_t section_offset;
} loom_native_object_symbol_layout_t;

typedef struct loom_native_object_fixup_t {
  // Section contribution containing the relocation site.
  iree_host_size_t section_contribution_index;
  // Byte offset of the relocation site within the section contribution.
  uint64_t section_offset;
  // Target-owned relocation kind identifier.
  uint32_t relocation_kind;
  // Symbol referenced by this fixup within the same object contribution.
  iree_host_size_t target_symbol_index;
  // Signed relocation addend.
  int64_t addend;
} loom_native_object_fixup_t;

typedef struct loom_native_object_fixup_layout_t {
  // Final assembled section containing the relocation site.
  iree_host_size_t section_index;
  // Byte offset of the relocation site within the final assembled section.
  uint64_t section_offset;
} loom_native_object_fixup_layout_t;

typedef struct loom_native_object_contribution_t {
  // Worker-produced section byte contributions.
  const loom_native_section_contribution_t* sections;
  // Number of worker-produced section byte contributions.
  iree_host_size_t section_count;
  // Worker-produced symbol definitions relative to |sections|.
  const loom_native_object_symbol_t* symbols;
  // Number of worker-produced symbol definitions.
  iree_host_size_t symbol_count;
  // Worker-produced relocation sites relative to |sections|.
  const loom_native_object_fixup_t* fixups;
  // Number of worker-produced relocation sites.
  iree_host_size_t fixup_count;
} loom_native_object_contribution_t;

// Resolves contribution-relative symbol positions into final section positions.
//
// |section_layouts| must be the layout rows returned by
// loom_native_assemble_section_contributions() for the same section
// contribution array referenced by |symbols|. The caller owns
// |out_symbol_layouts| and must provide at least |symbol_count| entries.
iree_status_t loom_native_object_resolve_symbol_layouts(
    const loom_native_object_symbol_t* symbols, iree_host_size_t symbol_count,
    const loom_native_section_contribution_layout_t* section_layouts,
    iree_host_size_t section_layout_count,
    loom_native_object_symbol_layout_t* out_symbol_layouts);

// Resolves contribution-relative fixup positions into final section positions.
//
// |symbol_count| is the number of symbols in the object contribution that owns
// |fixups|; each fixup target index must reference that symbol table.
iree_status_t loom_native_object_resolve_fixup_layouts(
    const loom_native_object_fixup_t* fixups, iree_host_size_t fixup_count,
    iree_host_size_t symbol_count,
    const loom_native_section_contribution_layout_t* section_layouts,
    iree_host_size_t section_layout_count,
    loom_native_object_fixup_layout_t* out_fixup_layouts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_OBJECT_H_
