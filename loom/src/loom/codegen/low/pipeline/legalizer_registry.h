// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Standard source legalizer registry composition for Low pipelines.

#ifndef LOOM_CODEGEN_LOW_PIPELINE_LEGALIZER_REGISTRY_H_
#define LOOM_CODEGEN_LOW_PIPELINE_LEGALIZER_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepares the immutable legalizer registry shared by pass executions.
//
// Target-specific providers are ordered before the generic buffer, scalar,
// vector, and view reference providers. The initialized storage must be
// deinitialized before |allocator| is released.
iree_status_t loom_low_legalizer_registry_storage_initialize(
    loom_target_legalizer_provider_list_t target_provider_list,
    iree_allocator_t allocator,
    loom_target_legalizer_registry_storage_t* out_storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PIPELINE_LEGALIZER_REGISTRY_H_
