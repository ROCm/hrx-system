// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Linker-owned semantic projections of symbol definitions.

#ifndef LOOM_LINK_SYMBOL_FACET_H_
#define LOOM_LINK_SYMBOL_FACET_H_

#include "iree/base/api.h"
#include "loom/ir/attribute_schema.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable semantic identity for one linker-owned symbol projection.
//
// Dialect-specific values use the defining dialect ID in the high byte and a
// manually assigned local ID in the low byte. Provider bytecode retains
// physical source-root indices; the module index translates those indices to
// these semantic identities before planning decisions are made.
typedef uint16_t loom_link_symbol_facet_kind_t;
enum loom_link_symbol_facet_kind_e {
  LOOM_LINK_SYMBOL_FACET_INVALID = 0,
  LOOM_LINK_SYMBOL_FACET_DEFINITION = 0x0001,
  LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT = 0x1001,
  LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION = 0x1002,
  LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION = 0x1003,
  LOOM_LINK_SYMBOL_FACET_COMMAND_CONTRACT = 0x1E01,
  LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION = 0x1E02,
};

// Compact source-role schema used to classify one indexed symbol's facets.
// This is derived once from existing structural interfaces and region roles;
// operation definitions do not carry linker metadata.
typedef struct loom_link_symbol_facet_schema_t {
  // Structural symbol interfaces declared by the defining operation.
  loom_symbol_interface_flags_t interfaces;
  // Number of root region slots declared by the defining operation.
  uint8_t root_region_count;
  // Function body source root region index plus one, or zero when absent.
  uint8_t body_region_index_plus_one;
  // Kernel configuration source root region index plus one, or zero when
  // absent.
  uint8_t kernel_configuration_region_index_plus_one;
  // Number of distinct semantic facets exposed by the symbol.
  uint8_t facet_count;
} loom_link_symbol_facet_schema_t;

static_assert(sizeof(loom_link_symbol_facet_schema_t) == 8,
              "link symbol facet schemas must remain 8 bytes");

// Classifies existing structural symbol metadata into a compact linker
// schema. Kernel and command interfaces currently own every root on their
// defining operations; adding an auxiliary root to either interface must
// update this classifier rather than creating another physical-root facet.
loom_link_symbol_facet_schema_t loom_link_symbol_facet_schema_classify(
    loom_symbol_interface_flags_t interfaces, uint8_t root_region_count,
    uint8_t body_region_index_plus_one,
    uint8_t kernel_configuration_region_index_plus_one);

// Returns semantic facet |ordinal| from |schema|, or INVALID when ordinal is
// out of range.
loom_link_symbol_facet_kind_t loom_link_symbol_facet_schema_kind_at(
    const loom_link_symbol_facet_schema_t* schema, uint8_t ordinal);

// Classifies a physical source root as one semantic facet, or INVALID when the
// source root is outside or unaccounted for by |schema|. Zero names the symbol
// contract; positive values name one-based root-region slots.
loom_link_symbol_facet_kind_t loom_link_symbol_facet_schema_source_root_kind(
    const loom_link_symbol_facet_schema_t* schema,
    uint8_t source_root_region_index_plus_one);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_SYMBOL_FACET_H_
