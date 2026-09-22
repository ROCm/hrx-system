// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/repr.h"

#include "loom/codegen/low/descriptor_traits.h"
#include "loom/codegen/low/immediates.h"
#include "loom/ops/low/ops.h"

static const loom_low_descriptor_registry_t* loom_low_repr_registry(
    const loom_low_repr_environment_state_t* state) {
  return (const loom_low_descriptor_registry_t*)state;
}

static const loom_low_descriptor_set_t* loom_low_repr_descriptor_set(
    const loom_low_repr_descriptor_set_t* descriptor_set) {
  return (const loom_low_descriptor_set_t*)descriptor_set;
}

static const loom_low_repr_descriptor_set_t*
loom_low_repr_lookup_descriptor_set_impl(
    const loom_low_repr_environment_state_t* state, iree_string_view_t key) {
  return (const loom_low_repr_descriptor_set_t*)
      loom_low_descriptor_registry_lookup(loom_low_repr_registry(state), key);
}

static bool loom_low_repr_resolve_descriptor_impl(
    const loom_low_repr_environment_state_t* state,
    const loom_low_repr_descriptor_set_t* descriptor_set_handle,
    iree_string_view_t key, loom_low_repr_descriptor_value_t* out_value) {
  (void)state;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_repr_descriptor_set(descriptor_set_handle);
  const uint32_t ordinal =
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key);
  if (ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return false;
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[ordinal];
  *out_value = (loom_low_repr_descriptor_value_t){
      .ordinal = ordinal,
      .effective_traits =
          loom_low_descriptor_effective_traits(descriptor_set, descriptor),
  };
  return true;
}

static iree_string_view_t loom_low_repr_descriptor_key_impl(
    const loom_low_repr_environment_state_t* state,
    const loom_low_repr_descriptor_set_t* descriptor_set_handle,
    uint32_t ordinal) {
  (void)state;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_repr_descriptor_set(descriptor_set_handle);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal);
  if (!descriptor) {
    return iree_string_view_empty();
  }
  return loom_low_descriptor_set_string(descriptor_set,
                                        descriptor->key_string_ref);
}

static iree_status_t loom_low_repr_resolve_packet_attributes_impl(
    const loom_low_repr_environment_state_t* state,
    const loom_low_repr_descriptor_set_t* descriptor_set_handle,
    loom_module_t* module, loom_op_t* op) {
  (void)state;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_repr_descriptor_set(descriptor_set_handle);
  const bool is_const = loom_low_const_isa(op);
  const uint32_t ordinal =
      is_const ? loom_low_const_descriptor(op) : loom_low_op_descriptor(op);
  loom_attribute_t attributes =
      is_const ? loom_low_const_attrs_attr(op) : loom_low_op_attrs_attr(op);
  IREE_RETURN_IF_ERROR(loom_low_resolve_immediate_enums(
      module, descriptor_set, &descriptor_set->descriptors[ordinal],
      &attributes));
  if (is_const) {
    loom_low_const_initialize_attrs(op, attributes);
  } else {
    loom_low_op_initialize_attrs(op, attributes);
  }
  return iree_ok_status();
}

static iree_string_view_t loom_low_repr_enum_value_token_impl(
    const loom_low_repr_environment_state_t* state,
    const loom_low_repr_descriptor_set_t* descriptor_set_handle,
    uint16_t domain_ordinal, int64_t value) {
  (void)state;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_repr_descriptor_set(descriptor_set_handle);
  const loom_low_enum_domain_t* domain =
      &descriptor_set->enum_domains[domain_ordinal];
  for (uint16_t i = 0; i < domain->value_count; ++i) {
    const loom_low_enum_value_t* entry =
        &descriptor_set->enum_values[domain->value_start + i];
    if (entry->value == value) {
      return loom_low_descriptor_set_string(descriptor_set,
                                            entry->token_string_ref);
    }
  }
  return iree_string_view_empty();
}

static const loom_low_repr_environment_vtable_t kLowReprEnvironmentVtable = {
    .lookup_descriptor_set = loom_low_repr_lookup_descriptor_set_impl,
    .resolve_descriptor = loom_low_repr_resolve_descriptor_impl,
    .descriptor_key = loom_low_repr_descriptor_key_impl,
    .resolve_packet_attributes = loom_low_repr_resolve_packet_attributes_impl,
    .enum_value_token = loom_low_repr_enum_value_token_impl,
};

void loom_low_repr_environment_initialize(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_low_repr_environment_t* out_environment) {
  *out_environment = (loom_low_repr_environment_t){
      .vtable = &kLowReprEnvironmentVtable,
      .state = (const loom_low_repr_environment_state_t*)descriptor_registry,
  };
}
