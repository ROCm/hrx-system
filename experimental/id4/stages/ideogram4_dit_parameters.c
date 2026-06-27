// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit_parameters.h"

#include <string.h>

static const iree_string_view_t kFp8E4m3LayerWeightSuffixes[] = {
    IREE_SVL("attention.qkv.weight"),   IREE_SVL("attention.o.weight"),
    IREE_SVL("feed_forward.w1.weight"), IREE_SVL("feed_forward.w3.weight"),
    IREE_SVL("feed_forward.w2.weight"), IREE_SVL("adaln_modulation.weight"),
};

static const iree_string_view_t kFp8E4m3ExactWeightKeys[] = {
    IREE_SVL("t_embedding.mlp_in.weight"),
    IREE_SVL("t_embedding.mlp_out.weight"),
    IREE_SVL("adaln_proj.weight"),
    IREE_SVL("input_proj.weight"),
    IREE_SVL("llm_cond_proj.weight"),
    IREE_SVL("final_layer.adaln_modulation.weight"),
    IREE_SVL("final_layer.linear.weight"),
};

iree_status_t id4_ideogram4_dit_parameter_format_parse(
    iree_string_view_t value,
    id4_ideogram4_dit_parameter_format_t* out_format) {
  IREE_ASSERT_ARGUMENT(out_format);
  if (iree_string_view_equal(value, IREE_SV("bf16"))) {
    *out_format = ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("fp8_e4m3"))) {
    *out_format = ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "DiT parameter format must be bf16 or fp8_e4m3");
}

iree_string_view_t id4_ideogram4_dit_parameter_format_name(
    id4_ideogram4_dit_parameter_format_t format) {
  switch (format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16:
      return IREE_SV("bf16");
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3:
      return IREE_SV("fp8_e4m3");
    default:
      return IREE_SV("invalid");
  }
}

static iree_status_t id4_ideogram4_dit_parameter_format_suffixes(
    id4_ideogram4_dit_parameter_format_t format,
    iree_string_view_list_t* out_suffixes) {
  IREE_ASSERT_ARGUMENT(out_suffixes);
  switch (format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3:
      *out_suffixes = (iree_string_view_list_t){
          .count = IREE_ARRAYSIZE(kFp8E4m3LayerWeightSuffixes),
          .values = kFp8E4m3LayerWeightSuffixes,
      };
      return iree_ok_status();
    default:
      *out_suffixes = iree_string_view_list_empty();
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid mixed DiT parameter format %" PRIu32,
                              (uint32_t)format);
  }
}

static iree_status_t id4_ideogram4_dit_parameter_format_exact_keys(
    id4_ideogram4_dit_parameter_format_t format,
    iree_string_view_list_t* out_exact_keys) {
  IREE_ASSERT_ARGUMENT(out_exact_keys);
  switch (format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3:
      *out_exact_keys = (iree_string_view_list_t){
          .count = IREE_ARRAYSIZE(kFp8E4m3ExactWeightKeys),
          .values = kFp8E4m3ExactWeightKeys,
      };
      return iree_ok_status();
    default:
      *out_exact_keys = iree_string_view_list_empty();
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid mixed DiT parameter format %" PRIu32,
                              (uint32_t)format);
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
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3:
      if (iree_string_view_is_empty(fp8_parameter_scope)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "FP8 e4m3 DiT parameter format requires an FP8 parameter scope");
      }
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid DiT parameter format %" PRIu32,
                              (uint32_t)format);
  }

  iree_string_view_list_t suffixes = iree_string_view_list_empty();
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_parameter_format_suffixes(format, &suffixes));
  iree_string_view_list_t exact_keys = iree_string_view_list_empty();
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_parameter_format_exact_keys(format, &exact_keys));
  const iree_host_size_t layer_rule_count = model.layer_count * suffixes.count;
  const iree_host_size_t rule_count = layer_rule_count + exact_keys.count;
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
  for (iree_host_size_t layer_ordinal = 0;
       layer_ordinal < model.layer_count && iree_status_is_ok(status);
       ++layer_ordinal) {
    for (iree_host_size_t suffix_ordinal = 0;
         suffix_ordinal < suffixes.count && iree_status_is_ok(status);
         ++suffix_ordinal) {
      const iree_host_size_t rule_ordinal =
          layer_ordinal * suffixes.count + suffix_ordinal;
      char* key_storage =
          out_rules->key_storage +
          rule_ordinal * ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY;
      iree_string_view_t key = iree_string_view_empty();
      status = id4_ideogram4_dit_program_format_layer_parameter(
          (uint32_t)layer_ordinal, suffixes.values[suffix_ordinal], key_storage,
          ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY, &key);
      if (iree_status_is_ok(status)) {
        out_rules->values[rule_ordinal] =
            (id4_ideogram4_dit_parameter_source_rule_t){
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
  }
  for (iree_host_size_t exact_key_ordinal = 0;
       exact_key_ordinal < exact_keys.count && iree_status_is_ok(status);
       ++exact_key_ordinal) {
    const iree_host_size_t rule_ordinal = layer_rule_count + exact_key_ordinal;
    out_rules->values[rule_ordinal] =
        (id4_ideogram4_dit_parameter_source_rule_t){
            // Exact logical parameter key.
            .key = exact_keys.values[exact_key_ordinal],
            // Scope containing the compact FP8 weight and row scale.
            .source_scope = fp8_parameter_scope,
            // Physical storage expected from the selected scope.
            .storage = ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED,
        };
    ++out_rules->count;
  }
  if (!iree_status_is_ok(status)) {
    id4_ideogram4_dit_parameter_source_rule_list_deinitialize(out_rules,
                                                              host_allocator);
  }
  return status;
}
