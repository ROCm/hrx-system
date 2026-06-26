// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PARAMETERS_H_
#define EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PARAMETERS_H_

#include "experimental/id4/stages/ideogram4_dit_program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// DiT parameter storage policy selected for Ideogram 4 DiT stages.
typedef enum id4_ideogram4_dit_parameter_format_e {
  // Invalid DiT parameter storage policy.
  ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID = 0,
  // All DiT parameters are sourced from BF16-expanded parameter scopes.
  ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16 = 1,
  // Supported DiT weights are sourced from native scaled FP8 scopes while
  // other DiT parameters remain sourced from BF16-expanded scopes.
  ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3 = 2,
} id4_ideogram4_dit_parameter_format_t;

// Owned exact-source rule list generated from one parameter format policy.
typedef struct id4_ideogram4_dit_parameter_source_rule_list_t {
  // Number of exact source rules in |values|.
  iree_host_size_t count;
  // Exact source rules generated for one DiT branch.
  id4_ideogram4_dit_parameter_source_rule_t* values;
  // Contiguous storage backing rule key string views.
  char* key_storage;
} id4_ideogram4_dit_parameter_source_rule_list_t;

// Parses a DiT parameter format flag value.
iree_status_t id4_ideogram4_dit_parameter_format_parse(
    iree_string_view_t value, id4_ideogram4_dit_parameter_format_t* out_format);

// Formats a DiT parameter format name.
iree_string_view_t id4_ideogram4_dit_parameter_format_name(
    id4_ideogram4_dit_parameter_format_t format);

// Initializes exact source rules for one DiT branch.
iree_status_t id4_ideogram4_dit_parameter_source_rule_list_initialize(
    id4_ideogram4_dit_parameter_format_t format,
    id4_ideogram4_dit_model_config_t model,
    iree_string_view_t fp8_parameter_scope, iree_allocator_t host_allocator,
    id4_ideogram4_dit_parameter_source_rule_list_t* out_rules);

// Deinitializes exact source rules initialized by
// id4_ideogram4_dit_parameter_source_rule_list_initialize.
void id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
    id4_ideogram4_dit_parameter_source_rule_list_t* rules,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PARAMETERS_H_
