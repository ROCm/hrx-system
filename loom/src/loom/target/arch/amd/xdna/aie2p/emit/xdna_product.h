// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical AIE2P XDNA ELF product emission.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_XDNA_PRODUCT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_XDNA_PRODUCT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "iree/schemas/xdna_executable.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/tile_link.h"
#include "loom/target/arch/amd/xdna/device/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

// One linked resident tile program and its physical placement.
typedef struct loom_aie2p_xdna_tile_t {
  // Physical compute tile executing the program.
  loom_xdna_tile_coordinate_t coordinate;
  // Diagnostic symbol name for this physical program instance.
  iree_string_view_t program_name;
  // Detached contribution from which symbols and sizes are retained.
  const loom_aie2p_leaf_contribution_t* contribution;
  // Fully placed and fixed-up native sections.
  const loom_aie2p_linked_tile_t* linked_tile;
} loom_aie2p_xdna_tile_t;

// One independently dispatchable array entry in an XDNA product.
typedef struct loom_aie2p_xdna_entry_t {
  // Diagnostic and runtime export name.
  iree_string_view_t name;
  // Required partition width, including worker-free service resources.
  uint16_t column_count;
  // Complete external requirements in binding-ordinal order. Extents include
  // control and service storage as well as payloads; commands do not imply
  // them.
  const iree_xdna_elf_binding_record_t* bindings;
  // Number of records in |bindings|.
  iree_host_size_t binding_count;
  // Typed array and invocation-control program awaiting final ordinals.
  const loom_aie2p_array_program_t* array_program;
  // Resident tile programs in worker order.
  const loom_aie2p_xdna_tile_t* tiles;
  // Number of records in |tiles|.
  iree_host_size_t tile_count;
} loom_aie2p_xdna_entry_t;

// Complete inputs to one canonical multi-entry AIE2P XDNA product.
typedef struct loom_aie2p_xdna_product_t {
  // Exact deployment profile serialized into image metadata.
  const loom_xdna_device_profile_t* device_profile;
  // Independently dispatchable entries in stable export-ordinal order.
  const loom_aie2p_xdna_entry_t* entries;
  // Number of records in |entries|.
  iree_host_size_t entry_count;
} loom_aie2p_xdna_product_t;

// One image-format constraint that rejected product construction.
typedef enum loom_aie2p_xdna_product_issue_kind_e {
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_NONE = 0,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_ENTRY_COUNT = 1,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_ENTRY_NAME_BYTE_LENGTH = 2,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_BINDING_RECORD_COUNT = 3,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_RELOCATION_RECORD_COUNT = 4,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_METADATA_BYTE_LENGTH = 5,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_PROGRAM_HEADER_COUNT = 6,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_SECTION_HEADER_COUNT = 7,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_NATIVE_COMMAND_BYTE_LENGTH = 8,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_NATIVE_OPERATION_COUNT = 9,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_SYMBOL_STRING_BYTE_LENGTH = 10,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_FILE_BYTE_LENGTH = 11,
  LOOM_AIE2P_XDNA_PRODUCT_ISSUE_SECTION_NAME_BYTE_LENGTH = 12,
} loom_aie2p_xdna_product_issue_kind_t;

// Structured semantic rejection returned to the compiler boundary.
typedef struct loom_aie2p_xdna_product_issue_t {
  // Violated image-format constraint, or NONE when construction succeeded.
  loom_aie2p_xdna_product_issue_kind_t kind;
  // Entry ordinal owning the violating fact, or UINT32_MAX for the product.
  uint32_t entry_ordinal;
  // Observed quantity.
  uint64_t actual;
  // Inclusive minimum accepted quantity.
  uint64_t minimum;
  // Inclusive maximum accepted quantity.
  uint64_t maximum;
} loom_aie2p_xdna_product_issue_t;

// Opaque arena-owned source admission retained through resident compilation.
typedef struct loom_aie2p_xdna_product_admission_t
    loom_aie2p_xdna_product_admission_t;

// Opaque arena-owned final image ready for serialization.
typedef struct loom_aie2p_xdna_product_image_t loom_aie2p_xdna_product_image_t;

// Admits all source-known product facts before resident compilation.
//
// |product| and its entry programs and binding records must remain valid
// through loom_aie2p_xdna_product_finalize. Resident tile arrays may be absent
// until then. Semantic rejection returns OK with |out_admitted| false and a
// populated |out_issue|. Allocation failure is the only status failure. The
// retained admission is allocated from |arena| and remains valid until it is
// reset.
iree_status_t loom_aie2p_xdna_product_admit(
    const loom_aie2p_xdna_product_t* product, iree_arena_allocator_t* arena,
    bool* out_admitted, loom_aie2p_xdna_product_admission_t** out_admission,
    loom_aie2p_xdna_product_issue_t* out_issue);

// Finalizes linked resident code into one serialization-ready XDNA image.
//
// Native command ranges splice shared linked sections and command fragments
// into final ELF backing. Identical resident sections are interned before the
// exact section-directory limit is applied. Semantic rejection returns OK with
// |out_finalized| false and a populated |out_issue|. Allocation failure is the
// only status failure. The image and all referenced payloads remain valid until
// the admission arena is reset.
iree_status_t loom_aie2p_xdna_product_finalize(
    loom_aie2p_xdna_product_admission_t* admission, bool* out_finalized,
    loom_aie2p_xdna_product_image_t** out_image,
    loom_aie2p_xdna_product_issue_t* out_issue);

// Writes one admitted and finalized ELF32LE `.xdna` image.
//
// Construction has already fixed every section, segment, payload, and file
// offset. Only output-stream failures remain externally reachable here.
iree_status_t loom_aie2p_xdna_product_write(
    const loom_aie2p_xdna_product_image_t* image, iree_io_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_XDNA_PRODUCT_H_
