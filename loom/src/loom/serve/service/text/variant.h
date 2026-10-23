// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_VARIANT_H_
#define LOOM_SERVE_SERVICE_TEXT_VARIANT_H_

#include "loom/serve/service/text/model.h"
#include "loom/serve/service/text/stages.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiles against immutable base declarations and parameter roots. This cold
// job never touches the owner's VM, execution timeline or retained row state.
// All borrowed base objects outlive completion. Source/sanitizer options are
// borrowed through completion. Rejection owes no callback.
iree_status_t loom_serve_text_variant_compile_async(
    loom_serve_text_model_t* owner, loom_serve_device_t* device,
    const loom_serve_preparation_t* preparation, iree_host_size_t shared_count,
    iree_host_size_t stage_count, const loom_serve_text_stage_t* baseline,
    iree_hal_command_buffer_mode_t command_mode,
    const loom_serve_sources_t* sources,
    const loomc_sanitizer_options_t* sanitizer,
    loom_serve_variant_completion_fn_t completion, void* user_data,
    iree_allocator_t host_allocator);

// Checks candidate ownership and the replacement private peak before rebinding
// native commands. The owner establishes the completed execution boundary and
// excludes peer admission until the new peak has been published.
iree_status_t loom_serve_text_variant_bind(
    const loom_serve_text_variant_t* variant, loom_serve_text_model_t* owner,
    iree_vm_module_t* module, loom_serve_residency_t* residency);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_VARIANT_H_
