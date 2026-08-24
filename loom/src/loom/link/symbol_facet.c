// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/symbol_facet.h"

#include "loom/ir/ir.h"

static_assert((LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT >> 8) ==
                  LOOM_DIALECT_KERNEL,
              "kernel facets must remain in the kernel dialect namespace");
static_assert((LOOM_LINK_SYMBOL_FACET_COMMAND_CONTRACT >> 8) ==
                  LOOM_DIALECT_COMMAND,
              "command facets must remain in the command dialect namespace");

typedef enum loom_link_symbol_facet_family_e {
  LOOM_LINK_SYMBOL_FACET_FAMILY_DEFINITION = 0,
  LOOM_LINK_SYMBOL_FACET_FAMILY_KERNEL = 1,
  LOOM_LINK_SYMBOL_FACET_FAMILY_COMMAND = 2,
} loom_link_symbol_facet_family_t;

// Returns the one semantic facet family selected by structural interfaces.
// This is the single policy point to extend when another symbol interface
// gains independently selectable linker projections.
static loom_link_symbol_facet_family_t loom_link_symbol_facet_family(
    loom_symbol_interface_flags_t interfaces) {
  const bool is_kernel =
      iree_any_bit_set(interfaces, LOOM_SYMBOL_INTERFACE_KERNEL);
  const bool is_command =
      iree_any_bit_set(interfaces, LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM);
  IREE_ASSERT(!(is_kernel && is_command));
  if (is_kernel) return LOOM_LINK_SYMBOL_FACET_FAMILY_KERNEL;
  if (is_command) return LOOM_LINK_SYMBOL_FACET_FAMILY_COMMAND;
  return LOOM_LINK_SYMBOL_FACET_FAMILY_DEFINITION;
}

loom_link_symbol_facet_schema_t loom_link_symbol_facet_schema_classify(
    loom_symbol_interface_flags_t interfaces, uint8_t root_region_count,
    uint8_t body_region_index_plus_one,
    uint8_t kernel_configuration_region_index_plus_one) {
  loom_link_symbol_facet_schema_t schema = {
      .interfaces = interfaces,
      .root_region_count = root_region_count,
      .body_region_index_plus_one = body_region_index_plus_one,
      .kernel_configuration_region_index_plus_one =
          kernel_configuration_region_index_plus_one,
      .facet_count = 1,
  };
  switch (loom_link_symbol_facet_family(interfaces)) {
    case LOOM_LINK_SYMBOL_FACET_FAMILY_KERNEL:
      IREE_ASSERT(!body_region_index_plus_one ||
                  body_region_index_plus_one !=
                      kernel_configuration_region_index_plus_one);
      schema.facet_count += kernel_configuration_region_index_plus_one != 0;
      schema.facet_count += body_region_index_plus_one != 0;
      IREE_ASSERT_EQ(schema.facet_count, root_region_count + 1);
      break;
    case LOOM_LINK_SYMBOL_FACET_FAMILY_COMMAND:
      IREE_ASSERT_EQ(kernel_configuration_region_index_plus_one, 0);
      schema.facet_count += body_region_index_plus_one != 0;
      IREE_ASSERT_EQ(schema.facet_count, root_region_count + 1);
      break;
    case LOOM_LINK_SYMBOL_FACET_FAMILY_DEFINITION:
      break;
  }
  return schema;
}

loom_link_symbol_facet_kind_t loom_link_symbol_facet_schema_kind_at(
    const loom_link_symbol_facet_schema_t* schema, uint8_t ordinal) {
  if (!schema || ordinal >= schema->facet_count) {
    return LOOM_LINK_SYMBOL_FACET_INVALID;
  }
  switch (loom_link_symbol_facet_family(schema->interfaces)) {
    case LOOM_LINK_SYMBOL_FACET_FAMILY_KERNEL: {
      uint8_t next_ordinal = 0;
      if (ordinal == next_ordinal++) {
        return LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT;
      }
      if (schema->kernel_configuration_region_index_plus_one != 0 &&
          ordinal == next_ordinal++) {
        return LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION;
      }
      if (schema->body_region_index_plus_one != 0 && ordinal == next_ordinal) {
        return LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION;
      }
      break;
    }
    case LOOM_LINK_SYMBOL_FACET_FAMILY_COMMAND:
      return ordinal == 0 ? LOOM_LINK_SYMBOL_FACET_COMMAND_CONTRACT
                          : LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION;
    case LOOM_LINK_SYMBOL_FACET_FAMILY_DEFINITION:
      return LOOM_LINK_SYMBOL_FACET_DEFINITION;
  }
  IREE_ASSERT_UNREACHABLE("facet ordinal is absent from its classified schema");
  return LOOM_LINK_SYMBOL_FACET_INVALID;
}

loom_link_symbol_facet_kind_t loom_link_symbol_facet_schema_source_root_kind(
    const loom_link_symbol_facet_schema_t* schema,
    uint8_t source_root_region_index_plus_one) {
  if (!schema ||
      source_root_region_index_plus_one > schema->root_region_count) {
    return LOOM_LINK_SYMBOL_FACET_INVALID;
  }
  if (source_root_region_index_plus_one == 0) {
    return loom_link_symbol_facet_schema_kind_at(schema, 0);
  }
  switch (loom_link_symbol_facet_family(schema->interfaces)) {
    case LOOM_LINK_SYMBOL_FACET_FAMILY_KERNEL:
      if (source_root_region_index_plus_one ==
          schema->kernel_configuration_region_index_plus_one) {
        return LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION;
      }
      if (source_root_region_index_plus_one ==
          schema->body_region_index_plus_one) {
        return LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION;
      }
      break;
    case LOOM_LINK_SYMBOL_FACET_FAMILY_COMMAND:
      if (source_root_region_index_plus_one ==
          schema->body_region_index_plus_one) {
        return LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION;
      }
      break;
    case LOOM_LINK_SYMBOL_FACET_FAMILY_DEFINITION:
      return LOOM_LINK_SYMBOL_FACET_DEFINITION;
  }
  IREE_ASSERT_UNREACHABLE("source root is absent from its classified schema");
  return LOOM_LINK_SYMBOL_FACET_INVALID;
}
