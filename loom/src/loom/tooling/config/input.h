// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-line and file adapters for Loom configuration bindings.

#ifndef LOOM_TOOLING_CONFIG_INPUT_H_
#define LOOM_TOOLING_CONFIG_INPUT_H_

#include "iree/base/api.h"
#include "loom/config/text_binding.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses and appends one `key=value` command-line assignment to |binding_set|.
//
// The split happens at the first '='. Both sides are trimmed. A leading '@' on
// the key is accepted and removed by the binding set so command-line spelling
// matches IR spelling without forcing shell users to quote sigils.
iree_status_t loom_tooling_config_text_binding_set_append_assignment(
    loom_config_text_binding_set_t* binding_set, iree_string_view_t assignment);

// Reads a JSON/JSONC config object file and appends flattened bindings.
//
// Empty paths and "-" are rejected. Command-line tools reserve those spellings
// for stdin/stdout, but config loading is a filesystem-path operation so
// accidental stdin consumption cannot race module input.
iree_status_t loom_tooling_config_text_binding_set_append_json_file(
    loom_config_text_binding_set_t* binding_set, iree_string_view_t path,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_CONFIG_INPUT_H_
