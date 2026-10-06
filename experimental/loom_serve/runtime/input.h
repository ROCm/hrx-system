// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_INPUT_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_INPUT_H_

#include "iree/tokenizer/tokenizer.h"
#include "iree/vm/environment.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Synchronous model-input capabilities, with no model policy or device access.
// The "input" module exports:
//   require(i32 condition, buffer diagnostic)
//   encode(buffer text, i32 flags, i64 capacity)
//       -> (buffer little_endian_ids, i64 count)
//   lookup(buffer token) -> i32 id
// encode retains at most capacity IDs and returns a zero-padded capacity*4 byte
// buffer plus its live count. Flags use iree_tokenizer_encode_flags_t:
// ADD_SPECIAL_TOKENS (4) enables tokenizer-defined framing, and
// NO_SPECIAL_TOKEN_MATCHING (8) encodes literal special-token spellings as
// ordinary text. Zero selects raw encoding with special-token matching. Every
// call starts a fresh input; AT_INPUT_START is implicit and TRACK_OFFSETS has
// no result channel in this ID-only capability. Other flag bits are rejected.
// A caller requiring the entire text requests one more ID than its limit and
// rejects a returned count above that limit instead of accepting truncation.
// Missing vocabulary tokens return -1. require propagates INVALID_ARGUMENT
// with the source diagnostic when condition is zero.
// The environment and optional tokenizer are borrowed for the module lifetime.
// A NULL tokenizer supports cold input validation only; encode/lookup then
// fail with FAILED_PRECONDITION. No request bytes are retained after a call.
// On failure *out_module is NULL.
iree_status_t loom_serve_input_module_create(iree_vm_environment_t* environment,
                                             const iree_tokenizer_t* tokenizer,
                                             iree_vm_module_t** out_module,
                                             iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_INPUT_H_
