// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PROGRAM_TARGET_H_
#define LOOM_RUNTIME_PROGRAM_TARGET_H_

#include "loom/runtime/program/compilation.h"
#include "loomc/target/iree_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiler and emission policy paired with one exact device loader target.
typedef struct loom_runtime_target_t {
  // Retained compiler profile and exact borrowed device loader target.
  loomc_iree_hal_target_selection_t selection;
  // Process-lifetime public emission format matching the loader target.
  iree_string_view_t artifact_format;
  // Source-to-low control-flow form required by the emitter.
  loomc_target_control_flow_lowering_t control_flow_lowering;
} loom_runtime_target_t;

// Selects from immutable device facts for the dispatch queue's affinity. No
// device URI or model modality participates in selection. Recognized but
// unsupported capabilities fail instead of trying another execution backend.
// On success the caller owns out_target->selection.target_profile; the
// executable target borrows device. The output is empty on failure.
// environment and providers are the exact compiler/runtime routes selected by
// the embedding application.
iree_status_t loom_runtime_target_select(
    loomc_target_environment_t* environment,
    const loomc_iree_hal_target_provider_t* const* providers,
    iree_host_size_t provider_count, iree_hal_device_t* device,
    iree_hal_physical_device_affinity_t affinity,
    loom_runtime_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_runtime_target_t* out_target);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_PROGRAM_TARGET_H_
