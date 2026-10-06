// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_RUNTIME_JSON_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_RUNTIME_JSON_H_

#include "iree/vm/environment.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates synchronous JSON utilities for source-owned request policy.
//
// json.members(buffer) -> (buffer, i32 type, i64 count) validates one complete
// UTF-8 object or array using IREE's JSON/JSONC parser. The owned result
// contains count records of five little-endian i64 fields: key offset/length,
// value offset/length, and type. Offsets refer to the input, with string quotes
// excluded and escapes preserved. Array keys have zero offset/length. Types
// are string=0, number=1, object=2, array=3, true=4, false=5, null=6. Input
// order and duplicate keys are preserved for the source policy to interpret.
//
// json.unescape(buffer source, i64 offset, i64 length, buffer target,
//               i64 target_offset) -> i64 length
// decodes a raw string range into the remaining target capacity. Source and
// destination must not overlap. Malformed escapes and insufficient storage
// fail; failure may leave a partially decoded prefix in the target, with no
// result length published. The operation allocates nothing and retains no
// caller storage. An empty target is not a size-query request.
//
// Request inputs and returned records are ordinary VM buffers. The environment
// outlives the module and every returned reference; the module owns no model,
// session, tokenizer, or asynchronous resource.
iree_status_t loom_serve_json_module_create(iree_vm_environment_t* environment,
                                            iree_vm_module_t** out_module,
                                            iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_RUNTIME_JSON_H_
