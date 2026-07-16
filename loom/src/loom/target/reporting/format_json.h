// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared compile report JSON value formatting.

#ifndef LOOM_TARGET_REPORTING_FORMAT_JSON_H_
#define LOOM_TARGET_REPORTING_FORMAT_JSON_H_

#include "loom/target/reporting/format.h"
#include "loom/util/json.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes an optional string field, using null when |value| is empty.
iree_status_t loom_target_compile_report_json_write_optional_string_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    iree_string_view_t value);

// Writes an optional u16 field, using null for UINT16_MAX.
iree_status_t loom_target_compile_report_json_write_optional_u16_field(
    loom_json_object_writer_t* object, iree_string_view_t name, uint16_t value);

// Writes an optional u64 field, using null for UINT64_MAX.
iree_status_t loom_target_compile_report_json_write_optional_u64_field(
    loom_json_object_writer_t* object, iree_string_view_t name, uint64_t value);

// Writes an optional u32 field, using null for UINT32_MAX.
iree_status_t loom_target_compile_report_json_write_optional_u32_field(
    loom_json_object_writer_t* object, iree_string_view_t name, uint32_t value);

// Writes the canonical static instruction-mix object.
iree_status_t loom_target_compile_report_format_instruction_mix_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    loom_output_stream_t* stream);

// Writes a u64 field only when |value| is nonzero.
iree_status_t loom_target_compile_report_json_write_nonzero_u64_field(
    loom_json_object_writer_t* object, iree_string_view_t key, uint64_t value);

// Writes one scaled memory-economics object.
iree_status_t loom_target_compile_report_format_memory_economics_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    uint64_t scale, loom_output_stream_t* stream);

// Writes one scaled operation-economics object.
iree_status_t loom_target_compile_report_format_operation_economics_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    uint64_t scale, loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_FORMAT_JSON_H_
