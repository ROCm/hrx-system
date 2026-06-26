// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit_parameters.h"

#include <string.h>

iree_status_t id4_ideogram4_dit_parameter_format_parse(
    iree_string_view_t value,
    id4_ideogram4_dit_parameter_format_t* out_format) {
  IREE_ASSERT_ARGUMENT(out_format);
  if (iree_string_view_equal(value, IREE_SV("bf16"))) {
    *out_format = ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("mixed_bf16_fp8_e4m3"))) {
    *out_format = ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "DiT parameter format must be bf16 or mixed_bf16_fp8_e4m3");
}

iree_string_view_t id4_ideogram4_dit_parameter_format_name(
    id4_ideogram4_dit_parameter_format_t format) {
  switch (format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16:
      return IREE_SV("bf16");
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3:
      return IREE_SV("mixed_bf16_fp8_e4m3");
    default:
      return IREE_SV("invalid");
  }
}

void id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
    id4_ideogram4_dit_parameter_source_rule_list_t* rules,
    iree_allocator_t host_allocator) {
  if (!rules) return;
  iree_allocator_free(host_allocator, rules->key_storage);
  iree_allocator_free(host_allocator, rules->values);
  memset(rules, 0, sizeof(*rules));
}

iree_status_t id4_ideogram4_dit_parameter_source_rule_list_initialize(
    id4_ideogram4_dit_parameter_format_t format,
    id4_ideogram4_dit_model_config_t model,
    iree_string_view_t fp8_parameter_scope, iree_allocator_t host_allocator,
    id4_ideogram4_dit_parameter_source_rule_list_t* out_rules) {
  IREE_ASSERT_ARGUMENT(out_rules);
  memset(out_rules, 0, sizeof(*out_rules));
  switch (format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16:
      return iree_ok_status();
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3:
      if (iree_string_view_is_empty(fp8_parameter_scope)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "mixed native-FP8 DiT parameter format requires an FP8 "
            "parameter scope");
      }
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid DiT parameter format %" PRIu32,
                              (uint32_t)format);
  }

  const iree_host_size_t rule_count = model.layer_count;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, rule_count, sizeof(out_rules->values[0]),
      (void**)&out_rules->values);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, rule_count,
        ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY,
        (void**)&out_rules->key_storage);
  }
  if (iree_status_is_ok(status)) {
    memset(out_rules->values, 0, rule_count * sizeof(out_rules->values[0]));
  }
  for (iree_host_size_t i = 0; i < rule_count && iree_status_is_ok(status);
       ++i) {
    char* key_storage = out_rules->key_storage +
                        i * ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY;
    iree_string_view_t key = iree_string_view_empty();
    status = id4_ideogram4_dit_program_format_layer_parameter(
        (uint32_t)i, IREE_SV("attention.qkv.weight"), key_storage,
        ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY, &key);
    if (iree_status_is_ok(status)) {
      out_rules->values[i] = (id4_ideogram4_dit_parameter_source_rule_t){
          // Exact logical parameter key.
          .key = key,
          // Scope containing the compact FP8 weight and row scale.
          .source_scope = fp8_parameter_scope,
          // Physical storage expected from the selected scope.
          .storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED,
      };
      ++out_rules->count;
    }
  }
  if (!iree_status_is_ok(status)) {
    id4_ideogram4_dit_parameter_source_rule_list_deinitialize(out_rules,
                                                              host_allocator);
  }
  return status;
}
