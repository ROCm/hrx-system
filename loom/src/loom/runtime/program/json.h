// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PROGRAM_JSON_H_
#define LOOM_RUNTIME_PROGRAM_JSON_H_

#include "iree/vm/environment.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates synchronous JSON utilities for source-owned request policy.
//
// json.compact(buffer) -> buffer validates one complete UTF-8 JSON/JSONC value
// and returns owned bytes without insignificant whitespace, comments or the
// leading BOM. String contents and escape spelling, number spelling, member
// order, duplicate keys and trailing commas are preserved. This is lexical
// compaction, not semantic canonicalization or conversion from JSONC to JSON.
//
// json.members(buffer) -> (buffer, i32 type, i64 count) validates one complete
// UTF-8 object or array using IREE's JSON/JSONC parser. The owned result
// contains count records of five little-endian i64 fields: key offset/length,
// value offset/length, and type. Offsets refer to the input, with string quotes
// excluded and escapes preserved. Array keys have zero offset/length. Types
// are string=0, number=1, object=2, array=3, true=4, false=5, null=6. Input
// order and duplicate keys are preserved for the source policy to interpret.
//
// json.members_range(buffer source, i64 offset, i64 length)
//     -> (buffer, i32 type, i64 count)
// has the same contract for one selected byte range. Returned offsets remain
// relative to source, so nested objects and arrays can be traversed directly
// from parent records without copying their JSON. Bytes outside the range are
// not inspected. The result owns its records and retains no source reference.
//
// json.unescape(buffer source, i64 offset, i64 length, buffer target,
//               i64 target_offset) -> i64 length
// decodes a raw string range into the remaining target capacity. Source and
// destination must not overlap. Malformed escapes and insufficient storage
// fail; failure may leave a partially decoded prefix in the target, with no
// result length published. The operation allocates nothing and retains no
// caller storage. An empty target is not a size-query request.
//
// json.parse_i64(buffer source, i64 offset, i64 length) -> i64
// json.parse_f64(buffer source, i64 offset, i64 length) -> f64
// parses one complete JSON number in the selected byte range. Integer parsing
// requires an integer spelling in the signed 64-bit range. Floating parsing
// rounds to finite F64; overflow and underflow to zero fail. Representable
// subnormals and signed zero are preserved. Source policy owns narrower
// application-specific ranges.
// These operations allocate nothing and retain no caller storage.
//
// json.format_f64(f64 value, i32 fractional_digits) -> buffer
// returns an owned JSON number in fixed decimal notation, using a period and
// exactly fractional_digits digits after it (0..17). Rounding is nearest-even
// from the represented binary value; integral digits and signed zero are
// preserved. Nonfinite values and unsupported precisions fail. The result has
// no quotes, whitespace, or trailing NUL and owns no caller storage.
//
// Request inputs and returned records are ordinary VM buffers. The environment
// outlives the module and every returned reference; the module owns no model,
// session, tokenizer, or asynchronous resource.
iree_status_t loom_runtime_json_module_create(
    iree_vm_environment_t* environment, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_PROGRAM_JSON_H_
