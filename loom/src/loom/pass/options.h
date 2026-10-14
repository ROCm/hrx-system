// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-driven decoding of textual and IR pass options.

#ifndef LOOM_PASS_OPTIONS_H_
#define LOOM_PASS_OPTIONS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/attribute.h"
#include "loom/pass/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decoded value for one pass option schema entry.
typedef struct loom_pass_decoded_option_t {
  // Option schema describing this value.
  const loom_pass_option_schema_t* schema;
  // True when the option was present in the source option dictionary.
  bool present;
  union {
    // String value when schema->kind is LOOM_PASS_OPTION_SCHEMA_STRING.
    iree_string_view_t string_value;
    // Integer value when schema->kind is LOOM_PASS_OPTION_SCHEMA_UINT32.
    uint32_t uint32_value;
    // Index into schema->enum_values when kind is LOOM_PASS_OPTION_SCHEMA_ENUM.
    uint16_t enum_value_index;
  };
} loom_pass_decoded_option_t;

// Immutable decoded option set for one pass descriptor.
typedef struct loom_pass_decoded_options_t {
  // Descriptor that owns the option schema.
  const loom_pass_descriptor_t* descriptor;
  // Arena-owned decoded options indexed by descriptor option schema order.
  const loom_pass_decoded_option_t* options;
  // Number of entries in |options|.
  uint16_t option_count;
} loom_pass_decoded_options_t;

// Validates textual option assignments against |descriptor|'s schema.
iree_status_t loom_pass_descriptor_validate_options(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t options);

// Decodes textual option assignments against |descriptor|'s schema into
// immutable arena-owned storage. String values reference the original option
// text and must not outlive it.
iree_status_t loom_pass_descriptor_decode_options(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t options,
    iree_arena_allocator_t* arena, loom_pass_decoded_options_t* out_options);

// Decodes a pass.run option attribute dictionary against |descriptor|'s schema
// into immutable arena-owned storage. String values are copied into the arena;
// enum values retain their descriptor-owned ordinal.
iree_status_t loom_pass_descriptor_decode_attr_options(
    const loom_pass_descriptor_t* descriptor, const loom_module_t* module,
    loom_named_attr_slice_t options, iree_arena_allocator_t* arena,
    loom_pass_decoded_options_t* out_options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_OPTIONS_H_
