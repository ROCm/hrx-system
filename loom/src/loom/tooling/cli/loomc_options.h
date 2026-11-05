// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-line option adapters for LoomC compiler invocations.

#ifndef LOOM_TOOLING_CLI_LOOMC_OPTIONS_H_
#define LOOM_TOOLING_CLI_LOOMC_OPTIONS_H_

#include "iree/base/api.h"
#include "loom/config/text_binding.h"
#include "loom/sanitizer/options.h"
#include "loomc/config.h"
#include "loomc/pass.h"
#include "loomc/sanitizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Adapts an owned command-line config set to borrowed LoomC options. The
// returned binding array is owned by |allocator| and must be freed by the
// caller. Binding strings continue to borrow from |text_set|.
iree_status_t loom_tooling_cli_make_loomc_config_options(
    const loom_config_text_binding_set_t* text_set, iree_allocator_t allocator,
    loomc_config_binding_t** out_bindings, loomc_config_options_t* out_options);

// Adapts parsed command-line sanitizer options to the LoomC descriptor used by
// compiler operations. The returned descriptor borrows no storage.
void loom_tooling_cli_make_loomc_sanitizer_options(
    const loom_sanitizer_options_t* options,
    loomc_sanitizer_options_t* out_options);

// Returns true when |pipeline| selects the target's default pass program.
// Empty and "default" values both select the default.
bool loom_tooling_cli_pipeline_uses_default(iree_string_view_t pipeline);

// Prepares the pass program selected by a command-line pipeline value.
//
// Empty and "default" values leave |out_pass_program| NULL so compilation can
// select the target's default program. "none" creates an explicit empty pass
// program, values beginning with '@' select a module-local pass.pipeline, and
// all other values are parsed as textual pass lists. |out_result| is populated
// only by selections that can produce structured compiler diagnostics.
iree_status_t loom_tooling_cli_prepare_loomc_pass_program(
    loomc_context_t* context, const loomc_module_t* module,
    iree_string_view_t pipeline, iree_string_view_t identifier,
    iree_allocator_t host_allocator, loomc_pass_program_t** out_pass_program,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_CLI_LOOMC_OPTIONS_H_
