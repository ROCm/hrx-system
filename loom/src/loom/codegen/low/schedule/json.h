// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Machine-readable JSON formatting for target-low schedule tables.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_JSON_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_JSON_H_

#include "iree/base/api.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/util/json.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes the common schedule hazard-gap fields into |object|. Callers may add
// context-specific identity fields before or after these fields.
iree_status_t loom_low_schedule_hazard_gap_write_json_fields(
    const loom_low_schedule_hazard_gap_t* hazard_gap,
    loom_json_object_writer_t* object);

// Appends a compact JSON object describing |table| to |builder|. The format
// is diagnostic/test output, not bytecode-stable artifact identity.
iree_status_t loom_low_schedule_format_json(
    const loom_low_schedule_table_t* table, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_JSON_H_
