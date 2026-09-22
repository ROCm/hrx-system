// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/immediates.h"

#include <string.h>

uint32_t loom_low_bind_immediate_presence(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_named_attr_slice_t attrs) {
  if (attrs.count == 0) {
    return 0;
  }
  if (attrs.count == descriptor->immediate_count) {
    return UINT32_MAX >> (32 - attrs.count);
  }
  uint32_t presence = 0;
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const iree_string_view_t name =
        loom_string_table_get(&module->strings, attrs.entries[i].name_id);
    for (uint16_t j = 0; j < descriptor->immediate_count; ++j) {
      const loom_low_immediate_t* immediate =
          &descriptor_set->immediates[descriptor->immediate_start + j];
      if (iree_string_view_equal(
              name, loom_low_descriptor_set_string(
                        descriptor_set, immediate->field_name_string_ref))) {
        presence |= immediate->attribute_mask;
        break;
      }
    }
  }
  return presence;
}

static bool loom_low_resolve_immediate_enum(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_string_view_t name,
    iree_string_view_t token, int64_t* out_value) {
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    if (immediate->kind != LOOM_LOW_IMMEDIATE_KIND_ENUM ||
        !iree_string_view_equal(
            name, loom_low_descriptor_set_string(
                      descriptor_set, immediate->field_name_string_ref))) {
      continue;
    }
    const loom_low_enum_domain_t* domain =
        &descriptor_set->enum_domains[immediate->enum_domain_id];
    for (uint16_t j = 0; j < domain->value_count; ++j) {
      const loom_low_enum_value_t* value =
          &descriptor_set->enum_values[domain->value_start + j];
      if (iree_string_view_equal(
              token, loom_low_descriptor_set_string(descriptor_set,
                                                    value->token_string_ref))) {
        *out_value = value->value;
        return true;
      }
    }
    return false;
  }
  return false;
}

iree_status_t loom_low_resolve_immediate_enums(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_attribute_t* attrs) {
  if (!iree_any_bit_set(descriptor->flags,
                        LOOM_LOW_DESCRIPTOR_FLAG_ENUM_IMMEDIATES) ||
      attrs->kind != LOOM_ATTR_DICT) {
    return iree_ok_status();
  }
  loom_named_attr_t* resolved_entries = NULL;
  for (uint16_t i = 0; i < attrs->count; ++i) {
    const loom_named_attr_t* entry = &attrs->dict_entries[i];
    if (entry->value.kind != LOOM_ATTR_STRING) {
      continue;
    }
    int64_t value = 0;
    if (!loom_low_resolve_immediate_enum(
            descriptor_set, descriptor,
            loom_string_table_get(&module->strings, entry->name_id),
            loom_string_table_get(&module->strings, entry->value.string_id),
            &value)) {
      continue;
    }
    if (resolved_entries == NULL) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &module->arena, attrs->count, sizeof(*resolved_entries),
          (void**)&resolved_entries));
      memcpy(resolved_entries, attrs->dict_entries,
             attrs->count * sizeof(*resolved_entries));
    }
    resolved_entries[i].value = loom_attr_i64(value);
  }
  if (resolved_entries != NULL) {
    *attrs = loom_make_canonical_attr_dict(resolved_entries, attrs->count);
  }
  return iree_ok_status();
}
