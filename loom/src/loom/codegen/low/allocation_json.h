// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Machine-readable JSON formatting for low allocation tables.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_JSON_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_JSON_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/allocation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends a compact JSON allocation table for |table| to |builder|. The
// JSON is intended for loom-check tests, diagnostics, and agent feedback; it is
// not the runtime allocation representation.
iree_status_t loom_low_allocation_format_json(
    const loom_low_allocation_table_t* table, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_JSON_H_
