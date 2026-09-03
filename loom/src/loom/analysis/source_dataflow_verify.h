// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Build-time verification for static source-dataflow provider tables.

#ifndef LOOM_ANALYSIS_SOURCE_DATAFLOW_VERIFY_H_
#define LOOM_ANALYSIS_SOURCE_DATAFLOW_VERIFY_H_

#include "iree/base/api.h"
#include "loom/analysis/source_dataflow.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies all static spans, references, and finite-domain invariants in
// |provider|. Target table tests call this once at build time; the production
// solver trusts verified static tables and does not rescan them per function.
iree_status_t loom_source_dataflow_provider_verify(
    const loom_source_dataflow_provider_t* provider);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SOURCE_DATAFLOW_VERIFY_H_
