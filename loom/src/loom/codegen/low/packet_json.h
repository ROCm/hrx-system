// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Machine-readable JSON formatting for scheduled target-low packet streams.

#ifndef LOOM_CODEGEN_LOW_PACKET_JSON_H_
#define LOOM_CODEGEN_LOW_PACKET_JSON_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends a compact JSON packet stream formed by joining |schedule| and
// |allocation| to |builder|. The output is a diagnostic/test format and a
// bytecode-shaped emitter input, not a durable serialized artifact.
iree_status_t loom_low_packet_format_json(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PACKET_JSON_H_
