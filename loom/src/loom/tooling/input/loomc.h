// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-line source admission through the public LoomC API.

#ifndef LOOM_TOOLING_INPUT_LOOMC_H_
#define LOOM_TOOLING_INPUT_LOOMC_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/tooling/input/format.h"
#include "loomc/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Imports one optional foreign source format through LoomC. Builtin Loom text
// and bytecode admission do not use this callback.
typedef iree_status_t (*loom_tooling_input_import_loomc_fn_t)(
    void* user_data, iree_string_view_t format,
    iree_string_view_t input_options,
    const loom_tooling_source_path_options_t* source_path_options,
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, loomc_module_t** out_module,
    loomc_result_t** out_result);

// One command-line source admission request. All views and providers are
// borrowed until admission returns.
typedef struct loom_tooling_loomc_input_options_t {
  // Resolved source format, provider options, and diagnostic path remapping.
  // input.format is the nonempty name returned by format/provider selection.
  loom_input_options_t input;
  // Physical path used for provider selection and relative include lookup.
  iree_string_view_t path;
  // Source bytes borrowed for the admission call.
  iree_string_view_t source;
  // Optional foreign-source importer dispatch.
  loom_tooling_input_import_loomc_fn_t import;
  // Opaque state forwarded to |import|.
  void* import_user_data;
} loom_tooling_loomc_input_options_t;

// Admits one source into a public LoomC module.
//
// Builtin Loom text and bytecode are deserialized directly. Other selected
// providers are dispatched through |options->import|. Source handles are
// invocation-local: returned modules and diagnostics own every retained source
// identity and byte snapshot required after this call returns. Physical paths
// are converted to their final logical identities during admission; later
// compiler operations and result printers must not apply the mapping again.
iree_status_t loom_tooling_input_admit_loomc_module(
    const loom_tooling_loomc_input_options_t* options, loomc_context_t* context,
    loomc_workspace_t* workspace, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, loomc_module_t** out_module,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_INPUT_LOOMC_H_
